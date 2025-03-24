"""
finetuning step 2
"""
import math
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
from torch.cuda.amp import autocast

import argparse
import torch
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from dataloader.loader_pairs import get_loader
from dataloader.preprocessing import augmentations_pretraining
from models.model_cl import RetCoPModel

from utils.constants import *
from utils.misc import set_seeds
from utils import misc

#########################################
#   model train
#########################################
def softce_clip_loss(model, logits_per_text, target_pseudo):
    caption_loss = ce_loss(model,logits_per_text, target_pseudo)
    image_loss = ce_loss(model,logits_per_text.T, target_pseudo)
    return (caption_loss + image_loss) / 2.0

def ce_loss(model, pred_logit, ref):
    ce_loss = torch.nn.functional.cross_entropy(pred_logit, ref)
    return ce_loss

def compute_logits(model, img_emb, text_emb):
    # similarity compute
    model.module.logit_scale.data = torch.clamp(model.module.logit_scale.data, 0, 4.6052)
    logit_scale = model.module.logit_scale.exp()
    logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
    return logits_per_text

def reduce_tensor(model, tensor: torch.Tensor):
    # loss  reduce loss
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= torch.distributed.get_world_size()
    return rt

#   model train main function
def fit(model, datalaoders, epochs=30, lr=5e-4, weight_decay=1e-5, scheduler=True, warmup_epoch=1, store_num=5,
        transforms=None, local_rank=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # tensorboard
    # if not os.path.isdir("./results/train_records"):
    #     os.mkdir("./results/train_records")
    # write = SummaryWriter(log_dir='../../local_data/results/train_records', flush_secs=60)

    # lr  scheduler
    if scheduler:
        # lr  linear warmup
        from utils.lr_scheduler import get_scheduler_per_iteration
        scheduler = get_scheduler_per_iteration(optimizer, lr, warmup_epoch, len(datalaoders["train"]))
    else:
        scheduler = None

    epoch = 1
    while epoch <= epochs:
        #   EK reference + mixed train
        loss_epoch = train_epoch_with_KD_loss_Atte_s(model,datalaoders["train"], optimizer, scheduler, transforms, epoch,
                                        datalaoders["KD"])
        if local_rank==0:
            print('Epoch=%d: ave_loss=%2.5f' % (epoch, loss_epoch))
            # write.add_scalar("train_loss", loss_epoch, epoch)

        #      save
        if (epoch % store_num == 0) & (local_rank==0):
            if model.module.out_path is not None:
                if not os.path.isdir(model.module.out_path):
                    os.mkdir(model.module.out_path)
                torch.save(model.module.state_dict(), model.module.out_path + model.module.vision_type + '_epoch' + str(epoch) + '.pth')
        epoch += 1



# ====================================================================================================

#   EK reference + mixed train
def train_epoch_with_KD_loss_Atte_s(model, loader, optimizer, scheduler=None, transforms=None, epoch=1, KD_loader=None):
    if loader is None:
        raise ValueError("dataloader is None")
    if KD_loader is None:
        raise ValueError("KD_loader is None")
    model.train()
    max_grad_norm, scaler = 1, torch.cuda.amp.GradScaler()                  #   amp
    loss_ave = 0.0

    loader.sampler.set_epoch(epoch)

    epoch_iterator = tqdm(loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=False)
    for step, (batch, KD_batch) in enumerate(zip(epoch_iterator, KD_loader)):
        images = batch["image"].to(model.device).to(torch.float32)
        text_tokens = model.module.text_model.tokenize(list(batch["report"][0]))    
        input_ids = text_tokens["input_ids"].to(model.device).to(torch.long)
        attention_mask = text_tokens["attention_mask"].to(model.device).to(torch.long)

        KD_images = KD_batch["image"].to(model.device).to(torch.float32)
        KD_text_tokens = model.module.text_model.tokenize(KD_batch["caption"])          # MM   MM data
        KD_input_ids = KD_text_tokens["input_ids"].to(model.device).to(torch.long)
        KD_attention_mask = KD_text_tokens["attention_mask"].to(model.device).to(torch.long)

        # In a batch, the corresponding category of a picture should be a positive sample with all texts of
        # the same category, and vice versa.
        coocurrence = np.array(
            [[iDesc == iiDesc for iDesc in batch["sel_category"]] for iiDesc in batch["sel_category"]], np.float32)
        target = torch.tensor(coocurrence / coocurrence.sum(-1)).to(model.device).to(torch.float32)       #   norm

        # MM  MM dataset
        KD_target = np.eye(len(KD_batch["caption"]), dtype=np.float32)
        KD_target = torch.tensor(KD_target).to(model.device).to(torch.float32)

        with autocast():                                                    # amp 
            print("\nExtracting features...")
            if transforms is not None:
                images = transforms(images)                                 # data augmentation  
            img_embeds = model.module.vision_model(images)
            text_embeds = model.module.text_model(input_ids, attention_mask)
            # print("img_embeds shape:", img_embeds.shape)  #  img_embeds 
            # print("text_embeds shape:", text_embeds.shape)  #  text_embeds 
            logits_per_text= compute_logits(model,img_embeds, text_embeds)   #   Similarity
            loss = softce_clip_loss(model,logits_per_text, target).to(model.device)# CLIP  clip loss

            print("\nExtracting KD features...")
            KD_img_embeds = model.module.vision_model(KD_images)
            KD_text_embeds = model.module.text_model(KD_input_ids, KD_attention_mask)

            KD_logits_per_text = compute_logits(model,KD_img_embeds, KD_text_embeds)  #    Similarity
            KD_loss = softce_clip_loss(model,KD_logits_per_text, KD_target).to(model.device)  # CLIP clip loss

            #   Similarity is calculated using the attention mechanism
            KD_embed = model.module.attention(img_embeds.unsqueeze(0), KD_img_embeds.unsqueeze(0), KD_text_embeds.unsqueeze(0)).squeeze(0)

            mse_loss = torch.nn.MSELoss()
            KD_norm_loss = mse_loss(text_embeds, KD_embed)
            loss = loss + KD_loss + KD_norm_loss * 100

            loss = reduce_tensor(model,loss)                                 # loss   Multithreaded loss merge

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        loss_ave += loss.item()
        torch.cuda.empty_cache()
        epoch_iterator.set_description(
            "Epoch=%d: Training (%d / %d Steps) " % (epoch, step + 1, len(loader)) +
            "- loss_value: " + str(round(loss.item(), 3))
        )

        #   iteration epoch
        # Learning rate scheduler. Here is a custom implementation of learning rate scheduler on iteration so in epoch for loop
        if scheduler is not None:
            scheduler.step()

    model.eval()
    return loss_ave / len(loader)


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
                             caption=args.caption, augment_description=args.augment_description, device=device,modality=args.modality)
    #   define model
    model = RetCoPModel(vision_type=args.architecture, out_path=args.out_path, from_checkpoint=args.load_weights, load_report=args.load_report, vision_pretrained=True,
                       weights_path=args.weights_path,weights_path_report=args.weights_path_report,local_rank=args.local_rank,device=device)
    model.to(device)
    #   parallelization
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                      find_unused_parameters=True)
    fit(model,dataloaders, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler,
              warmup_epoch=args.warmup_epoch, store_num=args.store_num, transforms=augmentations_pretraining,
              local_rank=args.local_rank)


def main():
    parser = argparse.ArgumentParser()

    #   Data
    parser.add_argument('--data_root_path', default=PATH_RESIZED_DATASETS)
    parser.add_argument('--dataframes_path', default=PATH_DATAFRAME_PRETRAIN_FFA)                         # pair  Pre-processed data pair
    parser.add_argument('--datasets', default=["FFA-IR"])
    parser.add_argument('--banned_categories', default=['myopia', 'cataract', 'macular hole', 'retinitis pigmentosa',
                                                        "myopic", "myope", "myop", "retinitis"])                   # OOD，  OOD experiment, delete corresponding category
    parser.add_argument('--out_path', default=PATH_RESULTS_PRETRAIN+"FT/", help='output path')
    parser.add_argument('--caption', default="A [ATR] fundus photograph of [CLS]")                    # prompt  prompt template
    parser.add_argument('--augment_description', default=True, type=lambda x: (str(x).lower() == 'true'))   #   Turn the category into a medical description?
    parser.add_argument('--balance', default=False, type=lambda x: (str(x).lower() == 'true'))        #  dataloader  data set balanced?
    parser.add_argument('--modality', default="CFP")            

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
    parser.add_argument('--load_report', default=True, type=lambda x: (str(x).lower() == 'true'))  #   load the pre-training weight？

    #   Device
    parser.add_argument('--store_num', default=1, type=int)                                         
    parser.add_argument('--num_workers', default=8, type=int, help='workers number for DataLoader')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank when using distributed training')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    args, unknown = parser.parse_known_args()
    process(args=args)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"                                                           
    main()