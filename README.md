# Prostate Cancer Detection from Whole Slide Images

Prostate cancer detection from whole slide images using Pytorch and Pytorch-lightning.

## Contents

- [Prostate Cancer Detection from Whole Slide Images](#prostate-cancer-detection-from-whole-slide-images)
  - [Contents](#contents)
  - [Problem](#problem)
  - [Data Processing](#data-processing)
  - [Training](#training)
    - [Model](#model)
    - [Loss](#loss)
  - [Results](#results)
  - [Winning Strategies](#winning-strategies)
  - [Reproducing](#reproducing)


## Problem

The aim of the [Prostate Cancer Grade Assessment Challenge](https://www.kaggle.com/c/prostate-cancer-grade-assessment) was to predict the ISUP grade level of prostate cancer present in whole slide images. The quadratic kappa metric was used to for evaluation in the competition so as to optimise for inter-rater agreement on the test-set labels. Detecting prostate cancer from whole slide images poses several challenges:

1. The images are extremely large but contain a small proportion of tissue per slide, making choosing relevant parts of the image important.
2. The data is fairly difficult to collect and so there are not too many examples (10616 in this competition, where around 2000 are different levels of the same images). Additionally, it suffers from significant variation in staining technique and equipment on a per-lab basis.
3. There is significant label noise as oncologists often can't agree. Additionally, this training set was labelled by students while the test set was labelled by a cohort of experts. Pen marks were also present in the training set but not in the test set.

## Data Processing

Processing the data is extremely important to select regions of the slide that are important to the classification while ignoring all of the background. I used 2 methods of selecting patches from the WSI's. The first method was a naive grid based method, generating a grid around the image and selecting patches based on the sum of the pixel values. This would select the darker patches of the cells while ignoring the white background. See naive_crop.py

![Naive crop generation](images/basic_patches.png)

The second method was more complicated in trying to select better patches from the WSI by applying a filter, thresholding, and then selecting the number of patches to take at successive y-coordinates along the image. Furthermore, this allows more control over the selection of patches: I could allow overlap or gaps between patches, choose how much a patch should be filled before it is included and offset the starting point of the patches so that the dataset could be augmented by patches that are translated across the slide. I also spent a lot of time improving the basic masking here by separating parts of the biopsy that were split by whitespace but in the same image, rotating these separate parts of the biopsy into a better position to avoid the background and filling gaps in the mask.

![Improved patch generation](images/betterconvcrop_masking_patches_improved_gaps.png)

Additionally, I tried a recursive tree based patching method which would recursively split the image into small patches, however, this didn't help. One more method which was interesting was a 'padding and alignment' method where I aligned all of the pixel rows in the images so to get a uniform image size for training the network. While the output was interesting, it didn't improve training and resulted in much slower training.

![pad align image](images/align-pad-rows-composite-2.png)

## Training

I trained my models using pytorch and pytorch-lightning, 42 patches of size 224. I also used half precision, gradient clipping and a OneCycle scheduler. I also tried model distilation on additional datasets from TCGA and PESO but this did not improve the results. See distill.py.

### Model

I used an Efficient-Net B0 model for most of my experiments (resnets and resnexts performed worse and took longer to train). I tried 2 variations on the same idea. First, I joined the patches up into a square image that was processed through the network with a basic regression objective. This worked well on the validation set but did not perform equally on the leaderboard.

The second variation passed each patch into the network individually, and the pooling them together in the head of the network. This method worked slightly worse in validation but performed better on the leaderboard. This conundrum led me to believe that the test set was quite different from the training (and validation) set, and thus led to the huge shake-up of the leaderboard at the end of the competition.

### Loss

I tried a few different losses: basic classification (CE), regression with MSE, MAE, and huber, and ordinal regression, as well as combining some together. Ordinal regression performed best on the validation and leaderboard in most of my experiments.

## Results

For the final submission, I ensembled 3 models with different patch sizes (160 and 224) and a dual loss (huber and ordinal). I used translation in the patch generation, TTA (horizontal and vertical flips and transpose), averaging together the predictions from all the models. I additionally submitted a model based on the naive generation as well, since this performed better on the leaderboard than the better patch generation.

| Model | Local Validation | Public LB | Private LB |
| ---   | ---              | ---       | ---        |
| Improved crop ensemble | Individually 0.87 | 0.88451 | 0.91042 |
| Naive crop model       | 0.89              | 0.89202 | 0.90514 |

The shake-up for this competition was particularly large - I was in the top 100 on the public LB but ended up at 485/1010 teams on the private LB. While the result is not particularly good, I learnt a lot in this competition and the result of 0.91 qk is still very good on such a task (the winning score was 0.94qk).

## Winning Strategies

Here I sum up some of the winning strategies from the top teams:

TODO


## Reproducing

First generate patches using scripts in `src/data_gen`.

Train:

```
pip install pytorch_lightning==0.7.6 efficientnet_pytorch timm kaggle albumentations
python train.py /path/to/patches --stack --npatches 42 --imsize 224
```
