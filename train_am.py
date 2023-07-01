import json
import cv2
import numpy as np

from torch.utils.data import Dataset
from io_utils import find_all_imgs, square_pad_resize
import os.path as osp
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers.wandb import WandbLogger
from cldm.model import create_model, load_state_dict
import random

import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
from einops import rearrange
import gc


class MyDataset(Dataset):
    def __init__(self, data_root: str, max_frame_step=4, prompt='', tgt_size=(640, 384)):
        framelist = find_all_imgs(data_root, abs_path=True)
        framelist.sort(key=lambda fname: int(osp.basename(fname).split('.')[0].split('_')[-1]))
        self.framelist = framelist
        self.tgt_step_list = list(range(-max_frame_step, max_frame_step+1))
        self.num_frame = len(self.framelist)
        self.prompt = prompt
        self.tgt_size = tgt_size
        self.max_frame_step = max_frame_step

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
        target = self.framelist[tgt_frame_idx]

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(cv2.imread(source), cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(cv2.imread(target), cv2.COLOR_BGR2RGB)
        source = cv2.resize(source, self.tgt_size, interpolation=cv2.INTER_LANCZOS4)
        target = cv2.resize(target, self.tgt_size, interpolation=cv2.INTER_LANCZOS4)


        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        temb = np.zeros((source.shape[0], source.shape[1], 1), dtype=np.float32)
        temb += tgt_step / self.max_frame_step
        source = np.concatenate((source, temb), axis=2)

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=self.prompt, hint=source)



class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, nimg=4):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.nimg = nimg

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    # def log_img(self, pl_module, batch, batch_idx, split="train"):
    #     check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
    #     if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
    #             hasattr(pl_module, "log_images") and
    #             callable(pl_module.log_images) and
    #             self.max_images > 0):
    #         logger = type(pl_module.logger)

    #         is_train = pl_module.training
    #         if is_train:
    #             pl_module.eval()

    #         with torch.no_grad():
    #             images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

    #         for k in images:
    #             N = min(images[k].shape[0], self.max_images)
    #             images[k] = images[k][:N]
    #             if isinstance(images[k], torch.Tensor):
    #                 images[k] = images[k].detach().cpu()
    #                 if self.clamp:
    #                     images[k] = torch.clamp(images[k], -1., 1.)

    #         self.log_local(pl_module.logger.save_dir, split, images,
    #                        pl_module.global_step, pl_module.current_epoch, batch_idx)

    #         if is_train:
    #             pl_module.train()


    @rank_zero_only
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = pl_module.global_step  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()
                

            with torch.no_grad():
                jpg = batch['jpg'][:self.nimg].clone()
                hint = batch['hint'][:self.nimg].clone()
                txt = batch['txt'][:self.nimg]
                del batch['jpg']
                del batch['hint']
                del batch['txt']
                del batch
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                images = pl_module.log_images({'jpg': jpg, 'hint': hint, 'txt': txt}, split=split, **self.log_images_kwargs)
                images.pop('conditioning')
            imglist = []
            images['control'] = images['control'][:, :3]
            for k in images:
                N = min(images[k].shape[0], self.max_images)
                img= images[k][:N]
                if isinstance(img, torch.Tensor):
                    img = img.detach().cpu()
                    if self.clamp:
                        img = torch.clamp(img, -1., 1.)
                    if self.rescale:
                        img = (img + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                    img = rearrange(img, 'b c h w -> (b h) w c')
                    # img = img.transpose(0, 1).transpose(1, 2).squeeze(-1)
                    img = img.numpy()
                    img = (img * 255).astype(np.uint8)
                    imglist.append(img)
            
            images = rearrange(imglist, '(row col) h w c -> (row h) (col w) c', col=len(images))
            pl_module.logger.log_image(key="samples", images=[images], step=batch_idx)
            # self.log_local(pl_module.logger.save_dir, split, images,
            #                pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()


    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")




# Configs
resume_path = './models/control_sd15_frame.ckpt'
batch_size = 4
logger_freq = 3000
learning_rate = 1e-5
sd_locked = True
only_mid_control = False
offline = False

from frametransdataset import FrameTransformDataset
# dataset = MyDataset(r'ama')
dataset = FrameTransformDataset('ama')
dataset.cache_transform()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)

logger = WandbLogger('cldm-tst', project='cldm', offline=offline)
imlogger = ImageLogger(batch_frequency=logger_freq, nimg=3)
# trainer = pl.Trainer(gpus=1, precision=32, callbacks=[imlogger])


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15_frame.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


devices = [0]
accelerator = 'gpu'
accumulate_grad_batches = 1

strategy = None
if devices and len(devices) > 1:
    strategy = "ddp"
    set_multi_processing(distributed=True)
if strategy is None:
    strategy = 'auto'

from pytorch_lightning.callbacks import ModelCheckpoint
checkpoint_callback = ModelCheckpoint(save_last=False, every_n_train_steps=logger_freq)


# Misc
logger = WandbLogger('cldm-tst', project='cldm')
trainer = pl.Trainer(precision=32, 
                    accumulate_grad_batches=accumulate_grad_batches,
                    callbacks=[imlogger, checkpoint_callback], 
                    sync_batchnorm=strategy=="ddp",
                    accelerator=accelerator,
                    logger=logger, devices=devices)

# Train!
trainer.fit(model, dataloader)