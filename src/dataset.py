import os
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import albumentations as albu
from albumentations.pytorch import ToTensorV2, ToTensor
import cv2
import re, math, random
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from functools import partial
import pandas as pd
from pathlib import Path
import numpy as np
from PIL import Image
import skimage.io



def get_transforms(imsize, train=True, local=True, n_patches=None, is_stack=False):
    if train:
        if local: # local transforms are applied randomly for each patch
            return albu.Compose([
                albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=90, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255)),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.Transpose(),
                albu.GaussNoise(p=0.2), # Super Slow!(?)
                albu.CoarseDropout(6, 20, 20, 2, p=0.3, fill_value=255),
                # albu.RandomGridShuffle((3,3), p=0.5),
                # albu.JpegCompression(quality_lower=70, p=0.3),

                # albu.Resize(imsize, imsize),
            ])
        else: #Global transforms are applied consistently across all patches
            return albu.Compose(([
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.Transpose(p=0.25),
            ] if not is_stack else []) + [
                albu.RandomGamma((90, 110), p=0.25),
                albu.RandomContrast(p=0.25, limit=0.1),
                albu.RandomBrightness(p=0.25, limit=0.1),
                albu.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=15, val_shift_limit=5, p=0.25),
                albu.Resize(imsize, imsize),
                albu.InvertImg(always_apply=True),
                # albu.Lambda(to_hsv, always_apply=True),
                # albu.Normalize(always_apply=True, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
                # albu.Normalize(always_apply=True, mean=[0.5, 0.2, 0.85], std=[0.3, 0.3, 0.15]), # HSV normalization
                # albu.Normalize(always_apply=True, mean=[0.81, 0.6, 0.73], std=[0.4, 0.51, 0.41]), # Basic normalization
                albu.Normalize(always_apply=True, mean=[1-0.85, 1-0.71, 1-0.80], std=[0.16, 0.27, 0.18]), # Invert Normalization
                ToTensor()
            ],
            additional_targets={f'image{i}':'image' for i in range(n_patches-1)} if is_stack else None)
    else:
        return albu.Compose([
            albu.Resize(imsize, imsize),
            albu.InvertImg(always_apply=True),
            # albu.Lambda(to_hsv, always_apply=True),
            # albu.Normalize(always_apply=True, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            # albu.Normalize(always_apply=True, mean=[0.5, 0.2, 0.85], std=[0.3, 0.3, 0.15]),
            # albu.Normalize(always_apply=True, mean=[0.81, 0.6, 0.73], std=[0.4, 0.51, 0.41]),
            albu.Normalize(always_apply=True, mean=[1-0.85, 1-0.71, 1-0.80], std=[0.16, 0.27, 0.18]),
            ToTensor()
        ],
        additional_targets={f'image{i}':'image' for i in range(n_patches-1)} if is_stack else None)

def show_batch(dataset, n=16, figsize=(12,12), random=True, is_stack=False):
    r = math.ceil(math.sqrt(n))
    axes = plt.subplots(r,r,figsize=figsize)[1].flatten()
    for i,ax in enumerate(axes):
        if i<n:
            im,label = dataset[np.random.randint(len(dataset))] if random else dataset[i]
            if is_stack:
                if isinstance(im, torch.Tensor):
                    im = im.permute(0,2,3,1).numpy()
                s=im.shape
                ni=int(math.sqrt(s[0]))
                im = im.reshape((ni,ni,*s[1:]))
                im = im.transpose((0,2,1,3,4))
                im = im.reshape((s[0]//ni*s[1], s[0]//ni*s[2], *s[3:]))

            if isinstance(im, torch.Tensor):
                from components.utils import denorm
                im = denorm(im, mean=[0.81, 0.6, 0.73], std=[0.4, 0.51, 0.41])
            ax.imshow(im)
            ax.set_title(f'{label}')
        ax.set_axis_off()

# TODO: add folds
class PandaDataset(torch.utils.data.Dataset):
    def __init__(self, path, patch_size=128, train=True, val_pct=0.10, stack=False,
                fold=5, n_patches=4, shuffle=False, pad_white=True, ordinal=False, norm_target=True):
        self.fold = fold
        self.train = train
        self.path = Path(path)
        self.n = math.ceil(math.sqrt(n_patches))
        self.global_tfms = get_transforms(patch_size if stack else patch_size*self.n, train=train, local=False, n_patches=n_patches, is_stack=stack)
        self.local_tfms = get_transforms(patch_size, train=train, local=True, n_patches=n_patches, is_stack=stack)
        self.n_patches = n_patches
        self.shuffle = shuffle
        self.pad_white = pad_white
        self.patch_size = patch_size
        self.norm_target = norm_target
        self.ordinal = ordinal
        self.stack = stack

        self.df = pd.read_json(path/'train.json')
        self.train_length = round((len(self.df))*(1-val_pct))
        # self.val_length = round((len(self.df))*(val_pct))
        self.df = self.df.iloc[:self.train_length] if self.train else self.df.iloc[self.train_length:]
        self.df = self.df.reset_index(drop=True)
        self.shape = cv2.imread(str(self.path/'images'/(self.df.patches[0][0]+'.jpg'))).shape

    def __len__(self):
        return len(self.df)
        # self.train_length if self.train else self.val_length

    def __getitem__(self, idx):
        # assert idx < self.train_length
        # if not self.train:
        #     assert idx < self.val_length
        #     idx += self.train_length
        df_row = self.df.iloc[idx]
        img_id = df_row.image_id
        img_label = df_row.isup_grade

        # Stack patches: #Do this before or after augmentation? Some augmentations before and some after: global ones like brightness and contrast after, local before: crop resize
        if len(df_row.patches) <= self.n_patches: # too few patches
            n = self.n_patches-len(df_row.patches)
            frames = df_row.patches + (list(np.random.choice(df_row.patches, n, replace=True)) if not self.pad_white else [None]*n)
            if self.shuffle: random.shuffle(frames)
        else: # too many patches
            frames = np.random.choice(df_row.patches, self.n_patches, replace=False)
            if not self.shuffle:
                idxs = np.array([int(i)*1000+int(j) for *_,i,j in [p.split('_')[1:] for p in frames]]).argsort()
                frames = frames[idxs]

        if self.train:
            patches = [(self.local_tfms(image=cv2.cvtColor(cv2.imread(str(self.path/'images'/(f+'.jpg'))), cv2.COLOR_BGR2RGB))['image'] if f is not None else np.ones(self.shape,dtype=np.uint8)*255) for f in frames]
        else:
            patches = [(cv2.cvtColor(cv2.imread(str(self.path/'images'/(f+'.jpg'))), cv2.COLOR_BGR2RGB) if f is not None else np.ones(self.shape,dtype=np.uint8)*255) for f in frames]

        if self.stack:
            ims = {'image'+str(i): im for i,im in enumerate(patches[:-1])}
            ims['image'] = patches[-1]
            image = torch.stack(list(self.global_tfms(**ims).values()))
        else:
            image = cv2.vconcat([cv2.hconcat(patches[i*self.n:(i+1)*self.n]) for i in range(self.n)])
            image = self.global_tfms(image=image)['image']

        if self.norm_target:
            target = torch.tensor(img_label-2.5, dtype=torch.float32)
        elif self.ordinal:
            target = torch.zeros(5, dtype=torch.float32)
            target[:img_label] = 1.
        else:
            target = torch.tensor(img_label, dtype=torch.float32)

        return image, target

def to_hsv(im, **params):
    return cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
