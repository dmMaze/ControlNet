import cv2
import numpy as np

from torch.utils.data import Dataset
from io_utils import find_all_imgs
import os.path as osp
import random
import os
from skimage.segmentation import slic
from pytorch_lightning.utilities.distributed import rank_zero_only
import imageio
import functools
from tqdm import tqdm

def cache_transform(cache_dir_name: str):
    
    def decorator(transform):
        
        @functools.wraps(transform)
        def wrapper(img: np.ndarray, imgname: str = None, cache: bool = False, cache_root: str = None, *args, **kwargs):
            
            if cache:
                cache_dir = osp.join(cache_root, cache_dir_name)
                if not osp.exists(cache_dir):
                    os.makedirs(cache_dir)
                cachep = osp.join(cache_dir, imgname)
                if osp.exists(cachep):
                    cached = imageio.imread(cachep)
                    if cached.shape[0] != img.shape[0] or cached.shape[1] != img.shape[1]:
                        cached = cv2.resize(cached, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                    return cached
                
            transformed = transform(img, *args, **kwargs)
            if cache:
                imageio.imwrite(cachep, transformed, quality=98)
            return transformed

        return wrapper

    return decorator

@cache_transform(cache_dir_name='grey')
def to_grey(img: np.ndarray, imgname: str = None, cache: bool = False, cache_root: str = None, *args, **kwargs):
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grey = grey[..., None].repeat(3, axis=2)
    return grey

@cache_transform(cache_dir_name='canny')
def canny(img: np.ndarray, imgname: str = None, cache: bool = False, cache_root: str = None, *args, **kwargs):
    # grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(img, 70, 140)
    canny = canny[..., None].repeat(3, axis=2)
    return canny

@cache_transform(cache_dir_name='superpixel')
def superpixel(img: np.ndarray, imgname: str = None, cache: bool = False, cache_root: str = None, *args, **kwargs):
    segments_slic = slic(img.astype(np.float32), n_segments=500, compactness=10, sigma=1,
                        start_label=0, convert2lab=True)
    rst = np.zeros_like(img)
    for ii in range(segments_slic.max() + 1):
        seeds = segments_slic == ii
        rst[seeds] = np.mean(img[seeds], axis=0)
    return rst

def build_transforms(transform_list):
    tlist = []
    for transdict in transform_list:
        trans_func = transdict['transform']
        cache = transdict['cache']
        if 'transform_kwargs' in transdict:
            transform_kwargs = transdict['transform_kwargs']
        else:
            transform_kwargs = dict()
        transform = lambda img, imgname, data_root: trans_func(img, imgname, cache, data_root, **transform_kwargs)
        tlist.append(transform)

    return tlist



class FrameTransformDataset(Dataset):
    def __init__(self, data_root: str, max_frame_step=4, prompt='', tgt_size=(640, 384)):
        
        framelist = find_all_imgs(data_root, abs_path=True)
        framelist.sort(key=lambda fname: int(osp.basename(fname).split('.')[0].split('_')[-1]))
        self.framelist = framelist
        self.tgt_step_list = list(range(-max_frame_step, max_frame_step+1))
        self.num_frame = len(self.framelist)
        self.prompt = prompt
        self.tgt_size = tgt_size
        self.max_frame_step = max_frame_step
        transform_list = [
            dict(transform=to_grey, cache=False),
            dict(transform=canny, cache=False),
            dict(transform=superpixel, cache=True)
        ]
        self.transform_list = transform_list
        self.data_root = data_root

    def random_transform(self, source: np.ndarray, target: np.ndarray, sourcename: str, targetname: str):
        
        transdict = random.choice(self.transform_list)

        trans_func = transdict['transform']
        cache = transdict['cache']
        if 'transform_kwargs' in transdict:
            transform_kwargs = transdict['transform_kwargs']
        else:
            transform_kwargs = dict()
        transform = lambda img, imgname, data_root: trans_func(img, imgname, cache, data_root, **transform_kwargs)

        invtransform = random.random() > 0.5
        if invtransform:
            source = np.concatenate((transform(target, targetname, self.data_root), source), axis=2)
        else:
            transform_exemplar = transform(source, sourcename, self.data_root)
            source = np.concatenate((target, transform_exemplar), axis=2)
            target = transform(target, targetname, self.data_root)
        return source, target


    def __len__(self):
        return len(self.framelist)

    def __getitem__(self, idx):
        tgt_step = random.choice(self.tgt_step_list)
        if idx == 0:
            tgt_step = abs(tgt_step)
        if idx == self.num_frame - 1:
            if tgt_step > 0:
                tgt_step = -tgt_step
        tgt_frame_idx = idx + tgt_step
        tgt_frame_idx = max(min(tgt_frame_idx, self.num_frame - 1), 0)

        source = self.framelist[idx]
        sourcename = osp.basename(source)
        target = self.framelist[tgt_frame_idx]
        targetname = osp.basename(target)

        # Do not forget that OpenCV read images in BGR order.
        source = imageio.imread(source)
        target = imageio.imread(target)
        source = cv2.resize(source, self.tgt_size, interpolation=cv2.INTER_LANCZOS4)
        target = cv2.resize(target, self.tgt_size, interpolation=cv2.INTER_LANCZOS4)

        source, target = self.random_transform(source, target, sourcename, targetname)

        source = source.astype(np.float32) / 255.0
        target = (target.astype(np.float32) / 127.5) - 1.0
        
        return dict(jpg=target, txt=self.prompt, hint=source)
    
    @rank_zero_only
    def cache_transform(self):
        for transdict in self.transform_list:
            if not transdict['cache']:
                continue
            if 'transform_kwargs' in transdict:
                transform_kwargs = transdict['transform_kwargs']
            else:
                transform_kwargs = dict()
            trans_func = transdict['transform']
            transform = lambda img, imgname, data_root: trans_func(img, imgname, True, data_root, **transform_kwargs)
            for ii in tqdm(range(len(self)), desc='caching transform...'):
                source = self.framelist[ii]
                sourcename = osp.basename(source)
                source = imageio.imread(source)
                transform(source, sourcename, self.data_root)

                


if __name__ == '__main__':
    ds = FrameTransformDataset(r'animedata/amatest')
    # ds.cache_transform()

    for bid in range(len(ds)):
        data = ds[bid]
        target = data['jpg']
        hint = data['hint']

        target = cv2.cvtColor(((target + 1) * 127.5).astype(np.uint8), cv2.COLOR_BGR2RGB)

        hint = (hint * 255).astype(np.uint8)
        hint1 = cv2.cvtColor(hint[..., :3], cv2.COLOR_BGR2RGB)

        cv2.imshow('target', target)
        cv2.imshow('source1', hint1)
        if hint.shape[-1] > 3:
            hint2 = cv2.cvtColor(hint[..., 3:], cv2.COLOR_BGR2RGB)
            cv2.imshow('source2', hint2)
        cv2.waitKey(0)