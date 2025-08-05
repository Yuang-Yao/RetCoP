"""
pretraining RetCoP step2
"""
import math
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
from torch.cuda.amp import autocast
import pandas as pd
import random
import argparse
import torch
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from dataloader.loader_pairs import get_loader, get_loader_cus1, get_loader_cus2
from dataloader.preprocessing import augmentations_pretraining
from models.model_cl import RetCoPModel

from utils.constants import *
from utils.misc import set_seeds
from utils import misc

from torch.utils.data import DataLoader, Subset, Sampler, Dataset
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data._utils.collate import default_collate
import torch.nn.functional as F
# import torchvision.transforms as trans  
from sklearn.cluster import KMeans, MiniBatchKMeans
import gc


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

def compute_logits_past(model, img_emb, text_emb):
    # similarity compute
    model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)
    logit_scale = model.logit_scale.exp()
    logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
    return logits_per_text

def reduce_tensor(model, tensor: torch.Tensor):
    # loss  reduce loss
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= torch.distributed.get_world_size()
    return rt

#   model train main function
def fit(model,model_past, datalaoders, epochs=30, lr=5e-4, weight_decay=1e-5, scheduler=True, warmup_epoch=1, store_num=5,
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
        loss_epoch = train_epoch_with_KD_loss_Atte_s(model,model_past,datalaoders["train"], optimizer, scheduler, transforms, epoch, datalaoders["KD"])
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


def train_epoch_with_KD_loss_Atte_s(model, model_past, loader, optimizer, scheduler=None, transforms=None, epoch=1, KD_loader=None):
    if loader is None:
        raise ValueError("dataloader is None")
    if KD_loader is None:
        raise ValueError("KD_loader is None")

    model.train()
    max_grad_norm, scaler = 1, torch.cuda.amp.GradScaler()  # 
    loss_ave = 0.0

    loader.sampler.set_epoch(epoch)

    epoch_iterator = tqdm(loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=False)
    for step, (batch, KD_batch) in enumerate(zip(epoch_iterator, KD_loader)):
        images = batch["image"].to(model.device).to(torch.float32)
        text_tokens = model.module.text_model.tokenize(list(batch["report"][0]))    
        input_ids = text_tokens["input_ids"].to(model.device).to(torch.long)
        attention_mask = text_tokens["attention_mask"].to(model.device).to(torch.long)

        KD_images = KD_batch["image"].to(model.device).to(torch.float32)
        KD_text_tokens = model.module.text_model.tokenize(KD_batch["caption"])          
        KD_input_ids = KD_text_tokens["input_ids"].to(model.device).to(torch.long)
        KD_attention_mask = KD_text_tokens["attention_mask"].to(model.device).to(torch.long)

        # 
        coocurrence = np.array(
            [[iDesc == iiDesc for iDesc in batch["sel_category"]] for iiDesc in batch["sel_category"]], np.float32)
        target = torch.tensor(coocurrence / coocurrence.sum(-1)).to(model.device).to(torch.float32)

        # KD
        KD_target = np.eye(len(KD_batch["caption"]), dtype=np.float32)
        KD_target = torch.tensor(KD_target).to(model.device).to(torch.float32)

        with autocast():  # amp 
            if transforms is not None:
                images = transforms(images)  # 

            img_embeds = model.module.vision_model(images)
            text_embeds = model.module.text_model(input_ids, attention_mask)

            logits_per_text = compute_logits(model, img_embeds, text_embeds)
            loss = softce_clip_loss(model, logits_per_text, target).to(model.device)
            print(f"Original loss value: {loss.item()}")  # loss

            # Extracting KD features
            KD_img_embeds = model.module.vision_model(KD_images)
            KD_text_embeds = model.module.text_model(KD_input_ids, KD_attention_mask)

            KD_logits_per_text = compute_logits(model, KD_img_embeds, KD_text_embeds)  
            KD_loss = softce_clip_loss(model, KD_logits_per_text, KD_target).to(model.device)
            print(f"KD loss value: {KD_loss.item()}")  # KD loss

            # KL （I2T）
            with torch.no_grad():
                past_img_embeds = model_past.vision_model(images)
                past_text_embeds = model_past.text_model(input_ids, attention_mask)

            past_image_logits = compute_logits_past(model_past, past_img_embeds, past_text_embeds)
            current_image_logits = logits_per_text
            
            for i in range(current_image_logits.size(0)):
                if torch.max(past_image_logits[i]) != past_image_logits[i, i]:  
                    past_image_logits[i] = current_image_logits[i]  

             
            past_image_logits = F.softmax(past_image_logits, dim=-1)
            current_image_logits = F.softmax(current_image_logits, dim=-1)

            
            if torch.any(past_image_logits <= 0) or torch.any(current_image_logits <= 0):
                print(f"past_image_logits contains non-positive values: {past_image_logits}")
                print(f"current_image_logits contains non-positive values: {current_image_logits}")
                continue
            
            #  I2T_KL_loss 
            I2T_KL_loss = -torch.mean(torch.sum(past_image_logits * (torch.log(current_image_logits) - torch.log(past_image_logits)), dim=-1))
            if I2T_KL_loss < 0:
                print(f"I2T KL loss is negative: {I2T_KL_loss.item()}")
                continue
            # T2I KL 
            current_text_logits = compute_logits(model, text_embeds, img_embeds)
            past_text_logits = compute_logits_past(model_past, past_text_embeds, img_embeds)

             
            for i in range(current_text_logits.size(0)):
                if torch.max(past_text_logits[i]) != past_text_logits[i, i]:   
                    past_text_logits[i] = current_text_logits[i]   

            
            past_text_logits = F.softmax(past_text_logits, dim=-1)
            current_text_logits = F.softmax(current_text_logits, dim=-1)
            
            if torch.any(past_text_logits <= 0) or torch.any(current_text_logits <= 0):
                print(f"past_text_logits contains non-positive values: {past_text_logits}")
                print(f"current_text_logits contains non-positive values: {current_text_logits}")
                continue

            #  T2I_KL_loss 
            T2I_KL_loss = -torch.mean(torch.sum(past_text_logits * (torch.log(current_text_logits) - torch.log(past_text_logits)), dim=-1))
            if T2I_KL_loss < 0:
                print(f"T2I KL loss is negative: {T2I_KL_loss.item()}")
                continue

            KD_embed = model.module.attention(img_embeds.unsqueeze(0), KD_img_embeds.unsqueeze(0), KD_text_embeds.unsqueeze(0)).squeeze(0)

            mse_loss = torch.nn.MSELoss()
            KD_norm_loss = mse_loss(text_embeds, KD_embed)
            print(f"KD norm loss value: {KD_norm_loss.item()}")  # KD norm loss


            loss_1 = loss + KD_loss + KD_norm_loss * 100
            if loss_1 < 1:
                loss = loss + KD_loss + KD_norm_loss * 100 + I2T_KL_loss + T2I_KL_loss  #  KL 
            else:
                loss = loss_1
            loss = reduce_tensor(model, loss)  # loss


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

        if scheduler is not None:
            scheduler.step()

        # delete 
        del images, text_tokens, input_ids, attention_mask, KD_images, KD_text_tokens, KD_input_ids, KD_attention_mask
        del coocurrence, target, KD_target, img_embeds, text_embeds, logits_per_text, loss, KD_img_embeds, KD_text_embeds, KD_logits_per_text, KD_loss
        del past_img_embeds, past_text_embeds, past_image_logits, current_image_logits, I2T_KL_loss, current_text_logits, past_text_logits, T2I_KL_loss, KD_embed
        

    model.eval()
    return loss_ave / len(loader)







class CombinedDataset(Dataset):
    def __init__(self, ffa_dataset, cfp_dataset, seed=None):
        self.ffa_dataset = ffa_dataset
        self.cfp_dataset = cfp_dataset

        self.mixed_indices = [(i, 'ffa') for i in range(len(ffa_dataset))] + \
                             [(i, 'cfp') for i in range(len(cfp_dataset))]
        
        if seed is not None:
            random.seed(seed)

        random.shuffle(self.mixed_indices)

    def __len__(self):
        return len(self.mixed_indices)

    def __getitem__(self, idx):
        index, dataset_type = self.mixed_indices[idx]
        if dataset_type == 'ffa':
            return self.ffa_dataset[index]
        else:
            return self.cfp_dataset[index]


def select_representative_samples(model, cfp_kd_data, ratio=0.1, device="cuda"):
    """
     k-means  CFP 
    :param model: -
    :param cfp_kd_data: CFP 
    :param ratio:  ( 10%)
    :param device: 
    :return:  k-means  CFP 
    """
    model.eval()
    cfp_features = []

    #  CFP 
    with torch.no_grad():
        for item in cfp_kd_data:
            image = torch.from_numpy(item['image']).to(device).unsqueeze(0).float()
            text = item['caption']

        
            image_embedding = model.vision_model(image)

            #  input_ids  attention_mask
            text_input_ids, text_attention_mask = model.preprocess_text0(text)
            
            text_embedding = model.text_model(text_input_ids, text_attention_mask)

            similarity = torch.cosine_similarity(image_embedding, text_embedding, dim=-1)

   
            joint_embedding = similarity.unsqueeze(-1) * image_embedding + (1 - similarity).unsqueeze(-1) * text_embedding

            cfp_features.append(joint_embedding.cpu().numpy())

            # del image, text, image_embedding, text_input_ids, text_attention_mask, text_embedding, similarity, joint_embedding
            # torch.cuda.empty_cache()  #  GPU 
            # gc.collect()  #  CPU 

    cfp_features = np.vstack(cfp_features)  #  (N, d)，N ，d 


    #  C
    num_samples = len(cfp_kd_data)
    num_clusters = max(1, int(0.01 * num_samples))  # 1% 
    print("num_clusters:",num_clusters)

    # K-means 
    kmeans = KMeans(n_clusters=num_clusters, init="k-means++", random_state=42)
    cluster_labels = kmeans.fit_predict(cfp_features)
    cluster_centers = kmeans.cluster_centers_


    selected_indices = []
    for c in range(num_clusters):
        cluster_indices = np.where(cluster_labels == c)[0]
        if len(cluster_indices) == 0:
            continue

        #  L2 
        distances = np.linalg.norm(cfp_features[cluster_indices] - cluster_centers[c], axis=1)


        num_select = max(1, int(0.1 * len(cluster_indices)))  # 10% 
        top_k_indices = cluster_indices[np.argsort(distances)[:num_select]]
        selected_indices.extend(top_k_indices)

    #  CFP 
    representative_cfp_kd_data = [cfp_kd_data[i] for i in selected_indices]

    # cfp_features 
    del cfp_features
    torch.cuda.empty_cache()  #  GPU 
    gc.collect()  #  CPU 

    return representative_cfp_kd_data




def select_representative_samples_train(model, cfp_train_data, ratio=0.1, device="cuda"):
    """
     k-means  CFP 
    :param model: -
    :param cfp_train_data: CFP 
    :param ratio:  ( 10%)
    :param device: 
    :return:  k-means  CFP 
    """
    model.eval()
    cfp_features = []

    
    batch_size = 512
    num_batches = (len(cfp_train_data) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Processing Batches"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(cfp_train_data))
        batch_data = cfp_train_data[batch_start:batch_end]

        batch_image_embeddings = []
        batch_text_embeddings = []

        
        with torch.no_grad():
            for item in batch_data:
                image = torch.from_numpy(item['image']).to(device).unsqueeze(0).float()
                text = item['report']

            
                image_embedding = model.vision_model(image)
                batch_image_embeddings.append(image_embedding.cpu().numpy())

                #  input_ids  attention_mask
                text_input_ids, text_attention_mask = model.preprocess_text0(text)
                
                text_embedding = model.text_model(text_input_ids, text_attention_mask)
                batch_text_embeddings.append(text_embedding.cpu().numpy())

                # del image, text, image_embedding, text_input_ids, text_attention_mask, text_embedding
                # torch.cuda.empty_cache()  #  GPU 
                # gc.collect()  #  CPU 

        batch_image_embeddings = np.vstack(batch_image_embeddings)
        batch_text_embeddings = np.vstack(batch_text_embeddings)

        batch_similarity = torch.cosine_similarity(torch.tensor(batch_image_embeddings, device=device),
                                                   torch.tensor(batch_text_embeddings, device=device),
                                                   dim=-1).cpu().numpy()

        batch_joint_embedding = batch_similarity[:, np.newaxis] * batch_image_embeddings + \
                                (1 - batch_similarity[:, np.newaxis]) * batch_text_embeddings

        cfp_features.append(batch_joint_embedding)

        # del batch_image_embeddings, batch_text_embeddings, batch_similarity, batch_joint_embedding
        # torch.cuda.empty_cache()  #  GPU 
        # gc.collect()  #  CPU 


    cfp_features = np.vstack(cfp_features)  #  (N, d)，N ，d 

    num_samples = len(cfp_train_data)
    num_clusters = max(1, int(0.01 * num_samples))  # 1% 
    print(f"Number of Clusters: {num_clusters}")

    # **MiniBatch K-means **
    mini_batch_kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=2048, random_state=42)
    cluster_labels = mini_batch_kmeans.fit_predict(cfp_features)
    cluster_centers = mini_batch_kmeans.cluster_centers_

    selected_indices = []
    for c in range(num_clusters):
        cluster_indices = np.where(cluster_labels == c)[0]
        if len(cluster_indices) == 0:
            continue

        #  L2 
        distances = np.linalg.norm(cfp_features[cluster_indices] - cluster_centers[c], axis=1)

        #  K 
        num_select = max(1, int(0.1 * len(cluster_indices)))  # 10% 
        top_k_indices = cluster_indices[np.argsort(distances)[:num_select]]
        selected_indices.extend(top_k_indices)

    # ** CFP **
    representative_cfp_kd_data = [cfp_train_data[i] for i in selected_indices]

    del cfp_features
    torch.cuda.empty_cache()  #  GPU 
    gc.collect()  #    CPU 

    return representative_cfp_kd_data



def process(args):
    misc.init_distributed_mode(args)
    print("distributed succeed")
    device = torch.device(args.device)
    
    seed = 42
    set_seeds(seed, use_cuda=True)


    #   define model
    model = RetCoPModel(vision_type=args.architecture, out_path=args.out_path, from_checkpoint=args.load_weights, load_report=args.load_report, vision_pretrained=True,
                       weights_path=args.weights_path,weights_path_report=args.weights_path_report,local_rank=args.local_rank,device=device)
    model_past = RetCoPModel(vision_type=args.architecture, out_path=args.out_path, from_checkpoint=args.load_weights, load_report=False, vision_pretrained=True,
                       weights_path=args.weights_path,local_rank=args.local_rank,device=device)
    model.to(device)
    #   parallelization
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                      find_unused_parameters=True)

    #  FFA 
    dataloaders_FFA = get_loader(dataframes_path=args.dataframes_path, data_root_path=args.data_root_path,
                                datasets=args.datasets, balance=args.balance, batch_size=args.batch_size,
                                num_workers=args.num_workers, banned_categories=args.banned_categories,
                                caption=args.caption, augment_description=args.augment_description, device=device, modality=args.modality)
    print("FFA loaded successfully!")

    #  CFP 
    dataloaders_CFP = get_loader(dataframes_path=args.dataframes_path_CFP, data_root_path=args.data_root_path,
                                datasets=args.datasets_CFP, balance=args.balance, batch_size=args.batch_size,
                                num_workers=args.num_workers, banned_categories=args.banned_categories,
                                caption=args.caption, augment_description=args.augment_description, device=device)
    print("CFP loaded successfully!")



    print("start loading mixed data")

    #  FFA 
    ffa_train_data = dataloaders_FFA["train"].dataset  #  FFA  train 
    cfp_train_data = dataloaders_CFP["train"].dataset  #  CFP  train 
    print("train_data loaded successfully!")
    ffa_kd_data = dataloaders_FFA["KD"].dataset  #  FFA  KD 
    cfp_kd_data = dataloaders_CFP["KD"].dataset  #  CFP  KD 
    print("kd_data loaded successfully!")

    #  CFP ， 10% 
    sample_ratio = 0.1  # 
    selected_cfp_kd_data = select_representative_samples(model_past, cfp_kd_data, ratio=sample_ratio, device=device)
    print("kd_data selected successfully!")
    selected_cfp_train_data = select_representative_samples_train(model_past, cfp_train_data, ratio=sample_ratio, device=device)
    print("train_data selected successfully!")

    # 
    combined_train_dataset = CombinedDataset(ffa_train_data, selected_cfp_train_data)  # train FFACFP
    print("train_data combined successfully!")

    combined_kd_dataset = CombinedDataset(ffa_kd_data, selected_cfp_kd_data)  # KD  FFA  CFP 
    print("kd_data combined successfully!")



    #  train DataLoader  get_loader 
    train_loader = get_loader_cus1(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        dataset=combined_train_dataset
    )
    print("train_loader created successfully!")

    #  KD DataLoader  get_loader 
    kd_loader = get_loader_cus2(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        dataset=combined_kd_dataset,
        train_loader=train_loader
    )
    print("kd_loader created successfully!")


    #  dataloaders
    dataloaders = {
        "train": train_loader,
        "KD": kd_loader
    }

    print("mixed data loaded successfully!")

    # 
    del ffa_train_data, cfp_train_data, ffa_kd_data, cfp_kd_data, selected_cfp_kd_data, selected_cfp_train_data, combined_train_dataset, combined_kd_dataset
    torch.cuda.empty_cache()
    gc.collect()

    fit(model,model_past,dataloaders, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler,
              warmup_epoch=args.warmup_epoch, store_num=args.store_num, transforms=augmentations_pretraining,
              local_rank=args.local_rank)


def main():
    parser = argparse.ArgumentParser()

    #   Data
    parser.add_argument('--data_root_path', default=PATH_RESIZED_DATASETS)
    parser.add_argument('--dataframes_path_CFP', default=PATH_DATAFRAME_PRETRAIN)                         # pair  Pre-processed data pair
    parser.add_argument('--dataframes_path', default=PATH_DATAFRAME_PRETRAIN_FFA)                         # pair  Pre-processed data pair
    parser.add_argument('--datasets', default=["FFA-IR"])
    parser.add_argument('--datasets_CFP', default=["01_EYEPACS", "03_IDRID", "04_RFMid",
                                               "06_DEN", "07_LAG", "08_ODIR", "09_PAPILA", "10_PARAGUAY",
                                               "11_STARE", "12_ARIA", "14_AGAR300", "15_APTOS", "16_FUND-OCT",
                                               "17_DiaRetDB1", "18_DRIONS-DB", "19_Drishti-GS1",
                                               "20_E-ophta", "21_G1020", "23_HRF", "24_ORIGA", "26_ROC",
                                               "27_BRSET", "28_OIA-DDR", "29_AIROGS", "30_SUSTech-SYSU", "31_JICHI",
                                               "32_CHAKSU", "33_DR1-2", "34_Cataract", "35_ScarDat"])
    # parser.add_argument('--datasets_CFP', default=["01_EYEPACS"])
    parser.add_argument('--banned_categories', default=['myopia', 'cataract', 'macular hole', 'retinitis pigmentosa',
                                                        "myopic", "myope", "myop", "retinitis"])                   # OOD，  OOD experiment, delete corresponding category
    parser.add_argument('--out_path', default=PATH_RESULTS_PRETRAIN+"RetCoP/", help='output path')
    parser.add_argument('--caption', default="A [ATR] fundus photograph of [CLS]")                    # prompt  prompt template
    parser.add_argument('--augment_description', default=True, type=lambda x: (str(x).lower() == 'true'))   # ？  Turn the category into a medical description?
    parser.add_argument('--balance', default=False, type=lambda x: (str(x).lower() == 'true'))        # ？ dataloader  data set balanced?
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
