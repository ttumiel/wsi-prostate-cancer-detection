import os
import cv2
import skimage.io
from tqdm.notebook import tqdm
import zipfile
import numpy as np

TRAIN = '../input/prostate-cancer-grade-assessment/train_images/'
MASKS = '../input/prostate-cancer-grade-assessment/train_label_masks/'
OUT_TRAIN = 'train.zip'
OUT_MASKS = 'masks.zip'
sz = 128
N = 64
LEVEL = 1

BLOCKS = 4
v = 1

def tile(img):
    result = []
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=255)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < N:
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]
    for i in range(len(img)):
        result.append({'img':img[i], 'idx':i})
    return result



x_tot,x2_tot = [],[]
names = np.array([name[:-5] for name in os.listdir(TRAIN)])
names.sort()
names = names[names != '3790f55cad63053e956fb73027179707']
names = names[(v-1)*(len(names)//BLOCKS):(v)*(len(names)//BLOCKS)]

with zipfile.ZipFile(OUT_TRAIN, 'w') as img_out:
    for name in tqdm(names):
        img = skimage.io.MultiImage(os.path.join(TRAIN,name+'.tiff'))[LEVEL]
        tiles = tile(img)
        for t in tiles:
            img,idx = t['img'],t['idx']
            x_tot.append((img/255.0).reshape(-1,3).mean(0))
            x2_tot.append(((img/255.0)**2).reshape(-1,3).mean(0))
            #if read with PIL RGB turns into BGR
            img = cv2.imencode('.jpg',cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1]
            img_out.writestr(f'{name}_{idx}.jpg', img)


#image stats
img_avr =  np.array(x_tot).mean(0)
img_std =  np.sqrt(np.array(x2_tot).mean(0) - img_avr**2)
print('mean:',img_avr, ', std:', np.sqrt(img_std))
