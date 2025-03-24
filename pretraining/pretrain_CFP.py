"""
step2   pretraining CFP
"""

import argparse
import torch
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from dataloader.loader_pairs import get_loader
from dataloader.preprocessing import augmentations_pretraining
from models.model import RetCoPModel

from utils.constants import *
from utils.misc import set_seeds
from utils import misc



def process(args):
    misc.init_distributed_mode(args)
    print("distributed succeed")
    device = torch.device(args.device)
    
    seed = 42
    set_seeds(seed, use_cuda=True)

    # dataloader         ｛"train": train_loader｝
    # Create dataloader; Data Preprocessing; Merge data set; Return {"train": train_loader}
    dataloaders = get_loader(dataframes_path=args.dataframes_path, data_root_path=args.data_root_path,
                             datasets=args.datasets, balance=args.balance, batch_size=args.batch_size,
                             num_workers=args.num_workers, banned_categories=args.banned_categories,
                             caption=args.caption, augment_description=args.augment_description, device=device)
    #   define model
    model = RetCoPModel(vision_type=args.architecture, out_path=args.out_path, from_checkpoint=args.load_weights, load_report=True, vision_pretrained=True,
                       weights_path=args.weights_path,weights_path_report=args.weights_path_report,local_rank=args.local_rank,device=device)
    model.to(device)
    #   parallelization
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                      find_unused_parameters=True)
    model.module.fit(dataloaders, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler,
              warmup_epoch=args.warmup_epoch, store_num=args.store_num, transforms=augmentations_pretraining,
              local_rank=args.local_rank)


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
    parser.add_argument('--out_path', default=PATH_RESULTS_PRETRAIN+"keepfit/", help='output path')
    parser.add_argument('--caption', default="A [ATR] fundus photograph of [CLS]")                    # prompt  prompt template
    parser.add_argument('--augment_description', default=True, type=lambda x: (str(x).lower() == 'true'))   #   Turn the category into a medical description?
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
    parser.add_argument('--weights_path', default='/mnt/data_ssd/yayao/CPT_retinal/RetCoP/checkpoint')                         #   model weight
    parser.add_argument('--weights_path_report', default='/mnt/data_ssd/yayao/CPT_retinal/RetCoP/checkpoint')                         #   model weight
    parser.add_argument('--load_weights', default=True, type=lambda x: (str(x).lower() == 'true'))  #   load the pre-training weight？

    #   Device
    parser.add_argument('--store_num', default=1, type=int)                                         # 
    parser.add_argument('--num_workers', default=8, type=int, help='workers number for DataLoader')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank when using distributed training')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    args, unknown = parser.parse_known_args()
    process(args=args)


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"                                                           
    main()