import os
import torch, math
from torch import nn
from torch.nn import functional as F
from argparse import ArgumentParser
import pytorch_lightning as pl
from pathlib import Path
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from functools import partial
from torch.utils.data import DataLoader
import multiprocessing as mp
import pandas as pd

from dataset import PandaDataset, get_transforms
from metrics import quadratic_kappa, accuracy
from model import EffNetPatch, EffNetConcat
from table_logger import TableLogger
from train import BaselineModel


# TODO: subclass from baseline model
class StudentModel(pl.LightningModule):
    def __init__(self, path, hparams=None, teacher=None, lr=1e-3, bs=128, imsize=128, reg=True,
                n_patches=9, is_stack=False, epochs=None, norm_output=True, ordinal=False):
        super().__init__()
        self.path = path
        self.lr = lr
        self.bs = bs
        self.imsize = imsize
        self.reg = reg
        self.n_patches = n_patches
        self.is_stack = is_stack
        self.epochs = epochs
        self.norm_output = norm_output
        self.ordinal = ordinal

        if reg:
            if is_stack:
                self.model = EffNetPatch(n=5 if ordinal else 1, n_patches=n_patches)
            else:
                self.model = EffNetConcat(n=5 if ordinal else 1, n_patches=n_patches)

            if ordinal:
                self.loss_fn = nn.BCEWithLogitsLoss()
                # Try Focal Loss
            else:
                # self.loss_fn = F.mse_loss
                self.loss_fn = nn.SmoothL1Loss()
                # self.loss_fn = nn.L1Loss()
        else:
            if is_stack:
                self.model = EffNetPatch(n=6, n_patches=n_patches)
            else:
                self.model = EffNetConcat(n=6, n_patches=n_patches)
            # self.loss_fn = LabelSmoothingLoss(6, 0.1)
            self.loss_fn = F.cross_entropy

        self.teacher = BaselineModel.load_from_checkpoint(teacher, map_location='cuda:0', path=path, bs=bs,
                            imsize=imsize, lr=lr, n_patches=n_patches, reg=True, is_stack=is_stack,
                            norm_output=norm_output, ordinal=ordinal).eval()
        self.teacher.apply(lambda m: m.requires_grad_(False))

        # log hparams
        tfms = {'local': str(get_transforms(imsize, train=True, local=True, is_stack=is_stack, n_patches=n_patches)),
                'global': str(get_transforms(imsize, train=True, local=False, is_stack=is_stack, n_patches=n_patches))}
        self.hparams = {'lr': lr, 'bs':bs, 'patch_size':imsize, 'tfms': tfms,
                        'n_patches': n_patches, 'loss_fn': self.loss_fn.__class__.__name__,
                        'is_stack': is_stack, 'path': str(self.path), 'ordinal': ordinal, 'norm_target': norm_output}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            self.teacher = self.teacher.eval()
            y_teacher = self.teacher(x).detach().squeeze(1).sigmoid()

        y_hat = self(x)

        if self.reg:
            y_hat = y_hat.squeeze(1)

        loss = self.loss_fn(y_hat, y_teacher)
        y_hat = y_hat.detach()

        if self.reg:
            if self.norm_output:
                y_hat = y_hat.add(2.5)
                y = y.add(2.5)
            elif self.ordinal:
                y_hat = y_hat.sigmoid().sum(1)
                y = y.sum(1)
            preds = y_hat.round()
        else:
            preds = y_hat.argmax(1)

        tensorboard_logs = {
            'train/lr': self.trainer.optimizers[0].param_groups[0]['lr'],
            'train/acc': accuracy(preds, y),
            'train/loss': loss,
            'train/qk': quadratic_kappa(preds, y)
        }
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).detach()

        if self.reg:
            y_hat = y_hat.squeeze(1)

        loss = self.loss_fn(y_hat, y).detach()

        if self.reg:
            if self.norm_output:
                y_hat = y_hat.add(2.5)
                y = y.add(2.5)
            elif self.ordinal:
                y_hat = y_hat.sigmoid().sum(1)
                y = y.sum(1)
            preds = y_hat.round_()
        else:
            preds = y_hat.argmax(1)

        return {'val_loss': loss, 'acc': accuracy(preds, y), 'preds': preds, 'y': y}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x['preds'] for x in outputs])
        ys = torch.cat([x['y'] for x in outputs])
        n = len(preds)

        karo_idxs = torch.tensor(self.trainer.val_dataloaders[0].dataset.df.data_provider == 'karolinska')[:n]
        rad_idxs = torch.tensor(self.trainer.val_dataloaders[0].dataset.df.data_provider == 'radboud')[:n]

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        avg_qk = quadratic_kappa(preds[karo_idxs | rad_idxs], ys[karo_idxs | rad_idxs])
        avg_karo_qk = quadratic_kappa(preds[karo_idxs], ys[karo_idxs])
        avg_rad_qk = quadratic_kappa(preds[rad_idxs], ys[rad_idxs])
        tensorboard_logs = {'val/loss': avg_loss, 'val/acc': avg_acc, 'val/qk': avg_qk,
                            'val/qk/karolinska': avg_karo_qk, 'val/qk/radboud': avg_rad_qk}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=5e-6)
        # sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=2, cooldown=1, min_lr=1e-6)
        # if self.epochs>10: sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.epochs)
        len_dl = len(self.train_dataloader())
        sched = torch.optim.lr_scheduler.CyclicLR(optim, self.lr/100, self.lr, math.floor(len_dl*0.2),
                math.ceil(len_dl*0.8), gamma=0.9999, mode='exp_range', cycle_momentum=False) # cant cycle momentum for Adam
        return [optim], [{'scheduler': sched,
                          'interval': 'step'}]

    def train_dataloader(self):
        return DataLoader(PandaDataset(
            self.path, patch_size=self.imsize, n_patches=self.n_patches,
            ordinal=self.ordinal, norm_target=self.norm_output, stack=self.is_stack
        ),
            batch_size=self.bs, num_workers=mp.cpu_count(), shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(PandaDataset(
            self.path, patch_size=self.imsize, n_patches=self.n_patches, stack=self.is_stack,
            train=False, ordinal=self.ordinal, norm_target=self.norm_output
        ),
            batch_size=self.bs*3, num_workers=mp.cpu_count(), pin_memory=True)


def set_grad(m, grad):
    if isinstance(m, (nn.Linear,nn.BatchNorm2d)): return
    if hasattr(m, 'weight'):
        for p in m.parameters(): p.requires_grad_(grad)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', default='data/naive256x36/', nargs='?')
    parser.add_argument('-b', type=int, help='Batch Size', default=8)
    parser.add_argument('--imsize', type=int, help='Patch size', default=224)
    parser.add_argument('--npatches', type=int, help='Number of patches', default=36)
    parser.add_argument('--restore', help='Restore a model from a ckpt')
    parser.add_argument('--teacher', help='Restore a teacher from a ckpt')
    parser.add_argument('--stack', action='store_true')
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-2)
    args = parser.parse_args()
    print(args)

    path = Path(args.path)

    model = StudentModel(path, teacher=args.teacher, bs=args.b, lr=args.lr, imsize=args.imsize, n_patches=args.npatches,
                reg=True, is_stack=args.stack, epochs=5, norm_output=False, ordinal=True)
    # model = BaselineModel.load_from_checkpoint('./TableLogger/version_15/epoch=13.ckpt', map_location='cuda:0',
    #                                           path=path, bs=14, imsize=200, lr=1e-3, n_patches=22, reg=True, is_stack=False)

    logger = TableLogger()
    checkpoint_callback = ModelCheckpoint(
            filepath=logger.path,
            save_top_k=3,
            verbose=True,
            monitor='val_loss',
            mode='min',
            prefix='part1'
    )

    kwargs = {
        'train_percent_check': 1.0,
        'accumulate_grad_batches': 1,
        'val_percent_check': 1.0,
        'gpus': 1,
        'weights_summary':'top',
        'precision': 16,
        'logger': logger,
        'checkpoint_callback': checkpoint_callback,
        'gradient_clip_val': 1,
    }

    if args.restore is None:
        # Train head and batchnorm layers
        model.model.m.apply(partial(set_grad, grad=False))
        trainer = pl.Trainer(max_epochs=3, **kwargs)
        trainer.fit(model)

        # Train full network
        model.model.m.apply(partial(set_grad, grad=True))
        model.epochs = 30
        model.lr = model.lr/15.
        trainer = pl.Trainer(max_epochs=15, **kwargs)
        trainer.fit(model)

        checkpoint_callback = ModelCheckpoint(
            filepath=logger.path,
            save_top_k=3,
            verbose=True,
            monitor='val_loss',
            mode='min',
            prefix='part2'
        )
        kwargs['checkpoint_callback'] = checkpoint_callback

        model.lr = model.lr/10.
        trainer = pl.Trainer(max_epochs=10, **kwargs)
        trainer.fit(model)

    else:
        model.epochs = 30
        trainer = pl.Trainer(resume_from_checkpoint=args.restore, **kwargs)
        trainer.fit(model)

    # Save final checkpoint
    trainer.save_checkpoint(logger.path/'final.ckpt')
