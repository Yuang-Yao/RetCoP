"""
pretraining report
"""

import argparse
import torch
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from dataloader.loader_report import get_loader_report
from models.model import RetCoPModel
from models.model_mlm import MLM_Model

from utils.constants import *
from utils import misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.lr_scheduler import adjust_learning_rate

import time
from typing import Iterable
import numpy as np
import random
import timm.optim.optim_factory as optim_factory
from torch.utils.tensorboard import SummaryWriter
import math
import datetime
import json



#    Random seed   reproduction experiment
def set_seeds(seed_value, use_cuda):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)      # GPU  MULTI-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt="{global_avg:.6f}"))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = {"data": samples[0].to(device, non_blocking=True), "text_labels": samples[1].to(device, non_blocking=True),
                    "mask_attention": samples[2].to(device, non_blocking=True), "modality": "text", "task": "step_1:report"}
     
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)


        with torch.cuda.amp.autocast():
            # ，
            outputs = model(samples.copy())
            
            # 
            assert isinstance(outputs, tuple) and len(outputs) == 4, "Model output should be a tuple with four elements."
            
            # 
            (loss, _), _, _, _ = outputs
            
            #  loss 
            assert isinstance(loss, torch.Tensor), "Loss must be a tensor."
            assert loss.dim() == 0, "Loss should be a scalar tensor."

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update("loss", loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update("lr", lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    global_avg_print = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    print("Averaged stats:", global_avg_print)
    return global_avg_print



def process(args):
    misc.init_distributed_mode(args)
    print("distributed succeed")
    device = torch.device(args.device)


    seed = 42
    set_seeds(seed, use_cuda=True)

    # dataloader         ｛"train": train_loader｝
    dataset_train = get_loader_report(data_path=args.data_root_path, split="train", transform=None, max_words=112)

                             

    #   define model
    model = MLM_Model()
    model.to(device)



    print(f"Number of samples in dataset: {len(dataset_train)}")
    print(f"Sample data: {dataset_train[0]}")

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        # print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # pin_memory=args.pin_mem,
        drop_last=True,
    )

    model_without_ddp = model
    weight_decay = float(args.weight_decay)  # 
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, betas=(0.9, 0.95),weight_decay=weight_decay)
    loss_scaler = NativeScaler()

    #
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.out_path and ((epoch + 1) % args.store_num == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.out_path and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.out_path, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



def main():
    parser = argparse.ArgumentParser()

    #   Data
    parser.add_argument('--data_root_path', default=PATH_RESIZED_DATASETS)
    parser.add_argument('--dataframes_path', default=PATH_DATAFRAME_PRETRAIN)                         # pair  Pre-processed data pair
    parser.add_argument('--datasets', default=["01_EYEPACS", "03_IDRID", "04_RFMid",
                                               "06_DEN", "07_LAG", "08_ODIR", "09_PAPILA", "10_PARAGUAY",
                                               "11_STARE", "12_ARIA", "14_AGAR300", "15_APTOS", "16_FUND-OCT",
                                               "17_DiaRetDB1", "18_DRIONS-DB", "19_Drishti-GS1",
                                               "20_E-ophta", "21_G1020", "23_HRF", "24_ORIGA", "26_ROC",
                                               "27_BRSET", "28_OIA-DDR", "29_AIROGS", "30_SUSTech-SYSU", "31_JICHI",
                                               "32_CHAKSU", "33_DR1-2", "34_Cataract", "35_ScarDat"])
    parser.add_argument('--banned_categories', default=['myopia', 'cataract', 'macular hole', 'retinitis pigmentosa',
                                                        "myopic", "myope", "myop", "retinitis"])                   # OOD，  OOD experiment, delete corresponding category
    parser.add_argument('--out_path', default=PATH_RESULTS_PRETRAIN, help='output path')
    parser.add_argument('--caption', default="A [ATR] fundus photograph of [CLS]")                    # prompt  prompt template
    parser.add_argument('--augment_description', default=True, type=lambda x: (str(x).lower() == 'true'))   #  Turn the category into a medical description?
    parser.add_argument('--balance', default=False, type=lambda x: (str(x).lower() == 'true'))        #  dataloader  data set balanced?

    #   model architecture
    parser.add_argument('--architecture', default='resnet_v2', help='resnet_v1 -- efficientnet')      # img tower

    #   Training hyperparameter
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    parser.add_argument('--scheduler', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--warmup_epoch', default=1, type=int, help='number of warmup epochs')
    parser.add_argument('--weights_path', default='./checkpoint')                         #   model weight
    parser.add_argument('--load_weights', default=False, type=lambda x: (str(x).lower() == 'true'))  #   load the pre-training weight？

    #   Device
    parser.add_argument('--store_num', default=1, type=int)                                         # 
    parser.add_argument('--num_workers', default=8, type=int, help='workers number for DataLoader')
    parser.add_argument("--local_rank", type=int, default=-1)

    # others
    parser.add_argument('--mask_ratio', default=0.75, help='mlm')
    parser.add_argument('--log_dir', default=None, help='log')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')

    args, unknown = parser.parse_known_args()
    process(args=args)


if __name__ == "__main__":
    main()