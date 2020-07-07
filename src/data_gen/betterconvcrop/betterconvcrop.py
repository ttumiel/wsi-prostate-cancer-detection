# Pytorch
import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F

# General
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

# standard lib
import math
import random
from pathlib import Path
from functools import partial

import skimage.io
from tqdm.notebook import tqdm
from scipy.ndimage import label, generate_binary_structure, find_objects
import cv2
from scipy import ndimage


def open_image(path, level=2):
    return skimage.io.MultiImage(str(path))[level]

def process_image(image):
    return torch.from_numpy(image).float().div_(255).neg_().add_(1.).permute(2,0,1).unsqueeze_(0)

def segment(image, ks=5, thresh=0.1):
    weight = torch.ones(1,3,ks,ks).float()/(3*ks**2)
    filt = F.conv2d(image, weight, padding=ks//2)
    return filt.squeeze()>thresh

def crop_to_mask(im, mask):
    c = np.nonzero(mask)
    top_left = np.min(c, axis=1)
    bottom_right = np.max(c, axis=1)
    out = im[top_left[0]:bottom_right[0],
                top_left[1]:bottom_right[1]]
    return out

def make_square(im, min_size=256, fill_color=(255,255,255)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

def crop_to_white(im):
    proc_im = process_image(im)
    mask = segment(proc_im)
    crop = crop_to_mask(im, mask.numpy())
    return crop

def show_im(data):
    if isinstance(data, Image.Image):
        return data
    if isinstance(data, np.ndarray):
        return Image.fromarray(data)
    if isinstance(data, torch.Tensor):
        return Image.fromarray(data.cpu().detach().numpy())

def separate_objects_in_image(im):
    mask = get_resized_mask(im)
    s = generate_binary_structure(2,2)
    labels, n = label(mask,structure=s)
    objs = find_objects(labels)
    return objs


def sliceRect(input_slide, rect):
    """
    Take a cv2 rectangle object and remove its contents from
    a source image.
    Credit: https://stackoverflow.com/a/48553593
    """
    width = int(rect[1][0])
    height = int(rect[1][1])
    box = cv2.boxPoints(rect)

    src_pts = box.astype("float32")
    dst_pts = np.array(
        [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    output_slide = cv2.warpPerspective(input_slide, M, (width, height), borderValue=(255,255,255))
    return output_slide

def getContourRect(im):
    mask = segment(process_image(im)).numpy().astype('uint8')
    c,h = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(np.concatenate(c))
    return sliceRect(im, rect)

def get_resized_mask(im, small_size=1000):
    h,w,_ = im.shape
    sh,sw = (int(w/3), int(h/3)) if h>3000 else (h,w)
    im_small = np.asarray(Image.fromarray(im).resize((sh,sw)))
    mask = ndimage.binary_fill_holes(segment(process_image(im_small)).numpy())
    mask = np.asarray(Image.fromarray(mask.astype(float)).resize((w,h)))
    return mask.astype(bool)


def count_empty(arr, empty_val=245, thresh=0.8):
    pct_empty = np.sum(arr>=empty_val)/arr.size
    return pct_empty>thresh


def patch_image(im_id, patch_fn, save_ims=False, save_path='./data', level=2):
    base_im = open_image(path/f'train_images/{im_id}.tiff', level=level)
    if max(base_im.shape[:2]) > 3*1450*(1 if level == 2 else 4 if level == 1 else 16): # reduce size of input
        base_im = np.asarray(Image.fromarray(base_im).resize((base_im.shape[1]//2, base_im.shape[0]//2), Image.BILINEAR))
        print("overlarge input, shrinking")

    objs = separate_objects_in_image(base_im)

    ims = []
    for obj_num,obj in enumerate(objs):
        sliced_im = base_im[obj]
        if sliced_im.size > (60**2 if level==2 else 180**2 if level==1 else 480**2)*3: # minimum number of elements should be 60x60
            im = getContourRect(sliced_im)
            patches = patch_fn(im)
            if save_ims:
                for patch_num,p in enumerate(patches):
                    if not count_empty(p):
                        Image.fromarray(p).save(f'{save_path}/{im_id}_{obj_num}_{patch_num}.jpg')
                        ims.append(f"{im_id}_{obj_num}_{patch_num}")
                    else:
                        print(f'Image not saved due to insufficient data: {np.sum(p==255)/p.size}')
            else:
                ims.append(patches)
    return ims


def from_sliding_window(im, patch_size=128, overlap=20, return_coords=False, return_centers=False, width_thresh=0.5, y_start_offset=0):
    h,w,_ = im.shape

    if w>h:
        im = im.transpose((1,0, 2))
        h,w = w,h

    mask = get_resized_mask(im)

    hm,wm = mask.shape
    assert h==hm and w==wm

    centers = []
    crops = []

    # Loop over y index
    for y in range(patch_size//2+y_start_offset, h, patch_size-overlap):
        # threshold to the top and bottom of the image
        y0 = max(min(max(0,y-patch_size//2), h-patch_size),0)
        y1 = y0+patch_size

        line = mask[(y0+y1)//2]
        line_mask = line>0
        if not(np.sum(line_mask)>20):
            continue

        start_changes = np.where(~np.concatenate([[False], line_mask[:-1]]) & line_mask)[0]
        end_changes = np.where(~np.concatenate([line_mask[1:], [False]]) & line_mask)[0]

        for s,e in zip(start_changes, end_changes):
            mask_len = e-s+1
            midpoint = s + mask_len//2
            width = mask_len/patch_size
            if width>0.4 and width<width_thresh: width = 1
            width = math.ceil(width) if (width%1)>width_thresh else math.floor(width)
    #         if width>1: print(width, y)
            for p in range(width):
                x0 = int(max(min(max(0,round(p*patch_size-patch_size*width/2+midpoint)), w-patch_size),0))
                x1 = x0+patch_size

                centers.append(((x0+x1)//2, (y0+y1)//2))
                if return_coords:
                    crops.append(((y0,y1),(x0,x1)))
                else:
                    if np.all(np.array(im.shape[:2])==patch_size):
                        crops.append(im[y0:y1, x0:x1])
                    else:
                        crops.append(np.asarray(make_square(Image.fromarray(im[y0:y1, x0:x1]), min_size=patch_size)))

    if return_centers:
        return centers
    return crops



if __name__ == '__main__':
    # mkdir -p ./data/conv_crop
    path = Path('../input/prostate-cancer-grade-assessment/')
    df = pd.read_csv(path/'train.csv')
    df = df[df.image_id != '3790f55cad63053e956fb73027179707']
    df.head(10)

    SIZE = 224
    OVERLAP = 0
    WIDTH_THRESH = 0.7

    im_ids = []
    isups = []
    ns = []
    for im_id,isup in tqdm(zip(df.image_id,df.isup_grade), total=len(df.image_id)):
        outs = patch_image(im_id, partial(from_sliding_window,patch_size=SIZE, width_thresh=WIDTH_THRESH, overlap=OVERLAP), level=1, save_ims=True, save_path='./data/conv_crop')
        ns.append((len(outs)))
        im_ids.extend(outs)
        isups.extend([isup]*len(outs))


    pd.DataFrame({'image_id': im_ids, 'isup_grade': isups}).to_csv('./data/conv_crop/train.csv', index=False)

    ns = np.array(ns)
    print(ns.min(), ns.max(), ns.mean(), ns.std())
    plt.hist(ns)

# tar -zcf data.tar.gz data --remove-files
