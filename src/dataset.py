import os
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import albumentations as albu
from albumentations.pytorch import ToTensorV2
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
                albu.ShiftScaleRotate(shift_limit=0.1025, scale_limit=0.1, rotate_limit=45, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255)),
                albu.Flip(p=0.75),
                # albu.CoarseDropout(10, 20, 20, 2, p=0.5, fill_value=255),
                # albu.GaussNoise(p=0.1), # Super Slow!
                # albu.RandomGridShuffle((3,3), p=0.5),
                # albu.JpegCompression(quality_lower=70, p=0.3),

                # albu.Resize(imsize, imsize),
            ])
        else: #Global transforms are applied consistently across all patches
            return albu.Compose([
                # albu.RandomGamma((80, 120), p=0.4),
                # albu.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=15, val_shift_limit=0, p=0.2),
                albu.RandomContrast(p=0.2, limit=0.15),
                albu.RandomBrightness(p=0.2, limit=0.1),
                albu.Resize(imsize, imsize),
                albu.Normalize(always_apply=True, mean=[0.81, 0.6, 0.73], std=[0.4, 0.51, 0.41]),
                ToTensorV2()
            ],
            additional_targets={f'image{i}':'image' for i in range(n_patches-1)} if is_stack else None)
    else:
        return albu.Compose([
            albu.Resize(imsize, imsize),
            albu.Normalize(always_apply=True, mean=[0.81, 0.6, 0.73], std=[0.4, 0.51, 0.41]),
            ToTensorV2()
        ])

# TODO: add folds
class PandaDatasetStack(torch.utils.data.Dataset):
    def __init__(self, path, patch_size=128, train=True, val_pct=0.20,
                tfms=None, tiff_level=2, fold=5, n_patches=4, provider=None):
        self.fold = fold
        self.train = train
        self.path = Path(path)
        self.global_tfms = get_transforms(patch_size, train=train, local=False, n_patches=n_patches, is_stack=True)
        self.local_tfms = get_transforms(patch_size, train=train, local=True, n_patches=n_patches, is_stack=True)
        self.n_patches = n_patches

        self.df = pd.read_json(path/'train.json')
        if provider is not None:
            self.df = self.df[self.df.data_provider == provider]
        self.train_length = round((len(self.df))*(1-val_pct))
        self.val_length = round((len(self.df))*(val_pct))

    def __len__(self):
        return self.train_length if self.train else self.val_length

    def __getitem__(self, idx):
        assert idx < self.train_length
        if not self.train:
            assert idx < self.val_length
            idx += self.train_length
        df_row = self.df.iloc[idx]
        img_id = df_row.image_id
        img_label = df_row.isup_grade

        # Stack patches: #Do this before or after augmentation? Some augmentations before and some after: global ones like brightness and contrast after, local before: crop resize
        if len(df_row.patches) < self.n_patches:
            frames = df_row.patches + list(np.random.choice(df_row.patches, self.n_patches-len(df_row.patches), replace=True))
            random.shuffle(frames)
        else:
            frames = np.random.choice(df_row.patches, self.n_patches, replace=False)

        if self.train:
            ims = {f"image{i}": self.local_tfms(image=cv2.cvtColor(cv2.imread(str(self.path/'conv_crop'/(img_id+'.jpg'))), cv2.COLOR_BGR2RGB))['image'] for i,img_id in enumerate(frames[1:])}
            ims['image']=self.local_tfms(image=cv2.cvtColor(cv2.imread(str(self.path/'conv_crop'/(frames[0]+'.jpg'))), cv2.COLOR_BGR2RGB))['image']
            patches = list(self.global_tfms(**ims).values())
        else:
            patches = [self.global_tfms(image=cv2.cvtColor(cv2.imread(str(self.path/'conv_crop'/(img_id+'.jpg'))), cv2.COLOR_BGR2RGB))['image'] for img_id in frames]

        image = np.stack(patches) # Stack patches into vector
        return image, torch.tensor(img_label-2.5, dtype=torch.float32)


class PandaDatasetConcat(torch.utils.data.Dataset):
    def __init__(self, path, patch_size=128, train=True, val_pct=0.20,
                tfms=None, tiff_level=2, fold=5, n_patches=4):
        self.fold = fold
        self.train = train
        self.path = Path(path)
        self.n = math.ceil(math.sqrt(n_patches))
        self.global_tfms = get_transforms(patch_size*self.n, train=train, local=False, n_patches=n_patches)
        self.local_tfms = get_transforms(patch_size, train=train, local=True, n_patches=n_patches)
        self.n_patches = n_patches

        self.tiff_level = tiff_level
        self.df = pd.read_json(path/'train.json')
        self.train_length = round((len(self.df))*(1-val_pct))
        self.val_length = round((len(self.df))*(val_pct))

    def __len__(self):
        return self.train_length if self.train else self.val_length

    def __getitem__(self, idx):
        assert idx < self.train_length
        if not self.train:
            assert idx < self.val_length
            idx += self.train_length
        df_row = self.df.iloc[idx]
        img_id = df_row.image_id
        img_label = df_row.isup_grade

        # Stack patches: #Do this before or after augmentation? Some augmentations before and some after: global ones like brightness and contrast after, local before: crop resize
        if len(df_row.patches) < self.n_patches:
            frames = df_row.patches + list(np.random.choice(df_row.patches, self.n_patches-len(df_row.patches), replace=True))
            random.shuffle(frames)
        else:
            frames = np.random.choice(df_row.patches, self.n_patches, replace=False)

        if self.train:
            patches = [self.local_tfms(image=cv2.cvtColor(cv2.imread(str(self.path/'conv_crop'/(f+'.jpg'))), cv2.COLOR_BGR2RGB))['image'] for f in frames]
        else:
            patches = [cv2.cvtColor(cv2.imread(str(self.path/'conv_crop'/(img_id+'.jpg'))), cv2.COLOR_BGR2RGB) for img_id in frames]

        image = cv2.hconcat([cv2.vconcat(patches[i*self.n:(i+1)*self.n]) for i in range(self.n)])
        image = self.global_tfms(image=image)['image']

        return image, torch.tensor(img_label-2.5, dtype=torch.float32)
