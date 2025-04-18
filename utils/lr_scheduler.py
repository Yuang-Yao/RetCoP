"""

define pre-training learning rate schedule mechanism
"""

import math
import warnings
import torch

from typing import List
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,                     # iteration  warmup times
        max_epochs: int,                        #   max warmup times
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,                   #   min learning rate
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    # lr   LRScheduler.step   Returns the current lr  will be called by LRScheduler.step()
    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.",
                UserWarning,
            )

        # 0iteration  0th iteration
        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        # iteration
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)  
            ]                                                                           #   increase linearly

        # iteration  the last iteration
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs

        #   CosineAnnealingLR
        elif (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) /
            (1 +math.cos(math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs)))
            * (group["lr"] - self.eta_min) + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        #   Linear preheating
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        #   cosine adjustment
        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]


# lr  lr schedule in pre-training
def get_scheduler_per_iteration(optimizer, lr, warmup_epoch, its_per_epoch):
    # its_per_epoch：epoch iteration  len(dataloader)
    # its_per_epoch: The number of each epoch iteration. The parameter passed in is len(dataloader)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epoch * its_per_epoch,max_epochs=100 * its_per_epoch,
                                              warmup_start_lr=lr / (warmup_epoch * its_per_epoch))
    return scheduler


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epoch:
        lr = args.lr * epoch / args.warmup_epoch 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epoch) / (args.epochs - args.warmup_epoch)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
