import os
import torch
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

from dataset import PandaDatasetStack, PandaDatasetConcat, get_transforms
from metrics import quadratic_kappa, accuracy
from model import EffNetPatch, EffNetConcat
from table_logger import TableLogger

class BaselineModel(pl.LightningModule):
    def __init__(self, path, lr=1e-3, bs=128, imsize=128, reg=True,
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
            self.model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=6)
            # self.loss_fn = LabelSmoothingLoss(6, 0.1)
            self.loss_fn = F.cross_entropy

        # log hparams
        self.hparams = {'lr': lr, 'bs':bs, 'patch_size':imsize,
                        'n_patches': n_patches, 'loss_fn': self.loss_fn.__class__.__name__,
                        'tfms': str(get_transforms(imsize, train=True, local=True, is_stack=is_stack)),
                        'is_stack': is_stack, 'path': str(self.path), 'ordinal': ordinal, 'norm_target': norm_output}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        if self.reg:
            y_hat = y_hat.squeeze(1)

        loss = self.loss_fn(y_hat, y)
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

        return {'val_loss': loss, 'acc': accuracy(preds, y), 'qk': quadratic_kappa(preds, y)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        avg_qk = torch.stack([x['qk'] for x in outputs]).mean()
        tensorboard_logs = {'val/loss': avg_loss, 'val/acc': avg_acc, 'val/qk': avg_qk}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=3e-6)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=2, cooldown=1, min_lr=1e-6)
        if self.epochs>10: sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.epochs)
        return [optim], [sched]

    def train_dataloader(self):
        DS = PandaDatasetStack if self.is_stack else PandaDatasetConcat
        return DataLoader(DS(self.path, patch_size=self.imsize, n_patches=self.n_patches,
                            ordinal=self.ordinal, norm_target=self.norm_output),
                          batch_size=self.bs, num_workers=mp.cpu_count()-1, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        DS = PandaDatasetStack if self.is_stack else PandaDatasetConcat
        return DataLoader(DS(self.path, patch_size=self.imsize, n_patches=self.n_patches,
                            train=False, ordinal=self.ordinal, norm_target=self.norm_output),
                          batch_size=self.bs*4, num_workers=mp.cpu_count()-1, pin_memory=True)


def set_grad(m, grad):
    if isinstance(m, (nn.Linear,nn.BatchNorm2d)): return
    if hasattr(m, 'weight'):
        for p in m.parameters(): p.requires_grad_(grad)




# TODO: add cli
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', default='data/naive256x36/', nargs='?')
    parser.add_argument('-b', type=int, help='Batch Size', default=8)
    parser.add_argument('--restore')
    args = parser.parse_args()

    path = Path(args.path)

    model = BaselineModel(path, bs=args.b, lr=3e-4, imsize=224, n_patches=36,
                reg=True, is_stack=False, epochs=5, norm_output=False, ordinal=True)
    # model = BaselineModel.load_from_checkpoint('./TableLogger/version_15/epoch=13.ckpt', map_location='cuda:0',
    #                                           path=path, bs=14, imsize=200, lr=1e-3, n_patches=22, reg=True, is_stack=False)

    logger = TableLogger()
    checkpoint_callback = ModelCheckpoint(
            filepath=logger.path,
            save_top_k=3,
            verbose=True,
            monitor='val_loss',
            mode='min',
            prefix=''
    )

    kwargs = {
        'train_percent_check': 1.0,
        'accumulate_grad_batches': 2,
        'val_percent_check': 1.0,
        'gpus': 1,
        'weights_summary':'top',
        'precision': 16,
        'logger': logger,
        'checkpoint_callback': checkpoint_callback,
    }

    if args.restore is None:
        # Train head and batchnorm layers
        model.model.m.apply(partial(set_grad, grad=False))
        trainer = pl.Trainer(max_epochs=1, **kwargs)
        trainer.fit(model)

        # Train full network
        model.model.m.apply(partial(set_grad, grad=True))
        model.epochs = 30
        # model.lr = model.lr/5.
        trainer = pl.Trainer(max_epochs=30, **kwargs)

    else:
        trainer = pl.Trainer(resume_from_checkpoint=args.restore, **kwargs)
        trainer.fit(model)

    # Save final checkpoint
    trainer.save_checkpoint('final.ckpt')
