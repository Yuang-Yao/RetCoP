"""
model main function
"""
import math
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path

from models.dictionary import definitions
from models.archi import VisionModel, TextModel0, ProjectionLayer, MultiHeadAttention

import torch
import torchvision
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoTokenizer, logging           # 

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

PATH_PRETRAINED_WEIGHTS = "/mnt/data_ssd/yayao/CPT_retinal/RetCoP/checkpoint/"
ID_FLAIR_RESNET_V1 = "flair_resnet.pth"
URL_ID_FLAIR_RESNET_V1 = "1l24_2IzwQdnaa034I0zcyDLs_zMujsbR"


#   Model initialization, architecture, training, inference
class RetCoPModel(torch.nn.Module):
    #########################################
    #   Model initialization module
    #########################################
    def __init__(self, vision_type='resnet_v1', bert_type='/mnt/data/yayao/CPT_data/emilyalsentzer/Bio_ClinicalBERT', vision_pretrained=True,
                 proj_dim=512, proj_bias=False, logit_scale_init_value=0.07, from_checkpoint=True, load_report=True, weights_path=None, weights_path_report=None,
                 out_path=None, image_size=512, caption="A fundus photograph of [CLS]", projection=True, norm_features=True,local_rank=0,device="cuda"):
        super().__init__()
        # 、      Input format, parameter output
        self.local_rank=local_rank
        self.image_size = image_size
        self.caption = caption                                  #   Prompt template for prediction phase
        self.from_checkpoint = from_checkpoint
        self.weights_path = weights_path
        self.weights_path_report = weights_path_report
        self.out_path = out_path
        
        # transferlogit      norm logit?
        self.projection = projection
        self.norm_features = norm_features
        self.proj_dim = proj_dim
        self.proj_bias = proj_bias

        # pretrain  model architecture
        self.vision_type = vision_type
        self.bert_type = bert_type
        self.vision_pretrained = vision_pretrained              # pretrain  pretrain?
        self.logit_scale_init_value = logit_scale_init_value    #   learnable scale parameter
        self.vision_model = VisionModel(vision_type=self.vision_type, pretrained=self.vision_pretrained,
                                        proj_dim=self.proj_dim, proj_bias=self.proj_bias, projection=self.projection,
                                        norm=self.norm_features)
        self.text_model = TextModel0(bert_type=self.bert_type, proj_dim=self.proj_dim, proj_bias=self.proj_bias,
                                    projection=self.projection, norm=self.norm_features)
        self.logit_scale = torch.nn.Parameter(torch.log(torch.tensor(1/self.logit_scale_init_value)))

        # mixed train with cross Attention 、keepfit with cross Attention feature fusion  +  +（）
        # self.atte_TD2KD = MultiHeadAttention(key_size=self.proj_dim, query_size=self.proj_dim, value_size=self.proj_dim,
        #                                     num_hiddens=self.proj_dim, num_heads=1, dropout=0.5)
        # self.atte_KD2TD = MultiHeadAttention(key_size=self.proj_dim, query_size=self.proj_dim, value_size=self.proj_dim,
        #                                     num_hiddens=self.proj_dim, num_heads=1, dropout=0.5)


        #    Domain knowledge reference
        self.attention = MultiHeadAttention(key_size=self.proj_dim, query_size=self.proj_dim, value_size=self.proj_dim,
                                            num_hiddens=self.proj_dim, num_heads=2, dropout=0.5)

        if from_checkpoint:
            self.load_from_pretrained(self.weights_path)

        if load_report:
            self.load_from_pretrained_report(self.weights_path_report)
        self.device=device
        self.to(self.device)


    # /  Downstream task adaptation read/download pre-trained model parameters
    def load_from_pretrained(self, weights_path=None):
        # 
        state_dict = torch.load(weights_path, map_location="cuda:0")

        # 
        model_state_dict = self.text_model.state_dict()

        # 
        loaded_params = []
        ignored_params = []

        # ，
        filtered_state_dict = {}

        for key in state_dict['model'].keys():
            if key.startswith("fused_encoder."):
                new_key = key[len("fused_encoder."):]  # 
            else:
                new_key = key

            if new_key in model_state_dict:
                loaded_params.append(new_key)
                filtered_state_dict[new_key] = state_dict['model'][key]  # 
            else:
                ignored_params.append(new_key)

        # 
        print('Loaded parameters:', loaded_params)
        print('Ignored parameters:', ignored_params)

        # 
        self.text_model.load_state_dict(filtered_state_dict, strict=False)
        print('load model weight from:', weights_path)

    # /  Downstream task adaptation read/download pre-trained model parameters
    def load_from_pretrained_report(self, weights_path_report=None):
        if weights_path_report is None:
            raise ValueError("weights_path_report is None. Please provide a valid path.")
    
        print(f"Loading weights from: {weights_path_report}")

        device = f"cuda:{self.local_rank}"  #  local_rank 
        state_dict = torch.load(weights_path_report, map_location=device)
        # state_dict = torch.load(weights_path_report, map_location="cuda:0")

        
        model_state_dict = self.text_model.state_dict()

        
        loaded_params = []
        ignored_params = []

        
        filtered_state_dict = {}

        for key in state_dict['model'].keys():
            if key.startswith("fused_encoder."):
                new_key = key[len("fused_encoder."):]  # 
            else:
                new_key = key

            if new_key in model_state_dict:
                loaded_params.append(new_key)
                filtered_state_dict[new_key] = state_dict['model'][key]  # 
            else:
                ignored_params.append(new_key)

        # 
        print('Loaded parameters:', loaded_params)
        print('Ignored parameters:', ignored_params)

        
        self.text_model.load_state_dict(filtered_state_dict, strict=False)
        print('load model weight from:', weights_path_report)


    #########################################
    #   model train
    #########################################
    def softce_clip_loss(self, logits_per_text, target_pseudo):
        caption_loss = self.ce_loss(logits_per_text, target_pseudo)
        image_loss = self.ce_loss(logits_per_text.T, target_pseudo)
        return (caption_loss + image_loss) / 2.0

    def ce_loss(self, pred_logit, ref):
        ce_loss = torch.nn.functional.cross_entropy(pred_logit, ref)
        return ce_loss

    def compute_logits(self, img_emb, text_emb):
        # similarity compute
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text

    def reduce_tensor(self, tensor: torch.Tensor):
        # loss  reduce loss
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        rt /= torch.distributed.get_world_size()
        return rt

    #   model train main function
    def fit(self, datalaoders, epochs=30, lr=5e-4, weight_decay=1e-5, scheduler=True, warmup_epoch=1, store_num=5,
            transforms=None, local_rank=None):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

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

        epoch = 15
        while epoch <= epochs:

            #   EK reference + mixed train
            loss_epoch = self.train_epoch_with_KD_loss_Atte_s(datalaoders["train"], optimizer, scheduler, transforms, epoch,
                                          datalaoders["KD"])
        

            if local_rank==0:
                print('Epoch=%d: ave_loss=%2.5f' % (epoch, loss_epoch))
                # write.add_scalar("train_loss", loss_epoch, epoch)

            #      save
            if (epoch % store_num == 0) & (local_rank==0):
                if self.out_path is not None:
                    if not os.path.isdir(self.out_path):
                        os.mkdir(self.out_path)
                    torch.save(self.state_dict(), self.out_path + self.vision_type + '_epoch' + str(epoch) + '.pth')
            epoch += 1

  
    # ====================================================================================================

    #   EK reference + mixed train
    def train_epoch_with_KD_loss_Atte_s(self, loader, optimizer, scheduler=None, transforms=None, epoch=1, KD_loader=None):
        if loader is None:
            raise ValueError("dataloader is None")
        if KD_loader is None:
            raise ValueError("KD_loader is None")
        self.train()
        max_grad_norm, scaler = 1, torch.cuda.amp.GradScaler()                  #   amp
        loss_ave = 0.0

        loader.sampler.set_epoch(epoch)

        epoch_iterator = tqdm(loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=False)
        for step, (batch, KD_batch) in enumerate(zip(epoch_iterator, KD_loader)):
            images = batch["image"].to(self.device).to(torch.float32)
            text_tokens = self.text_model.tokenize(list(batch["report"][0]))    
            input_ids = text_tokens["input_ids"].to(self.device).to(torch.long)
            attention_mask = text_tokens["attention_mask"].to(self.device).to(torch.long)

            KD_images = KD_batch["image"].to(self.device).to(torch.float32)
            KD_text_tokens = self.text_model.tokenize(KD_batch["caption"])          # MM   MM data
            # KD_text_tokens = self.text_model.tokenize(list(KD_batch["report"][0]))  #  flair  baidu/flair data
            KD_input_ids = KD_text_tokens["input_ids"].to(self.device).to(torch.long)
            KD_attention_mask = KD_text_tokens["attention_mask"].to(self.device).to(torch.long)

            # In a batch, the corresponding category of a picture should be a positive sample with all texts of
            # the same category, and vice versa.
            coocurrence = np.array(
                [[iDesc == iiDesc for iDesc in batch["sel_category"]] for iiDesc in batch["sel_category"]], np.float32)
            target = torch.tensor(coocurrence / coocurrence.sum(-1)).to(self.device).to(torch.float32)       #   norm

            # MM  MM dataset
            KD_target = np.eye(len(KD_batch["caption"]), dtype=np.float32)
            KD_target = torch.tensor(KD_target).to(self.device).to(torch.float32)

            # flair   flair/baidu dataset
            # coocurrence = np.array(
            #     [[iDesc == iiDesc for iDesc in KD_batch["sel_category"]] for iiDesc in KD_batch["sel_category"]], np.float32)
            # KD_target = torch.tensor(coocurrence / coocurrence.sum(-1)).to(device).to(torch.float32)  # norm  

            with autocast():                                                    # amp 
                print("\nExtracting features...")
                if transforms is not None:
                    images = transforms(images)                                 # data augmentation  
                img_embeds = self.vision_model(images)
                text_embeds = self.text_model(input_ids, attention_mask)
                print("img_embeds shape:", img_embeds.shape)  #  img_embeds 
                print("text_embeds shape:", text_embeds.shape)  #  text_embeds 
                logits_per_text= self.compute_logits(img_embeds, text_embeds)   #   Similarity
                loss = self.softce_clip_loss(logits_per_text, target).to(self.device)# CLIP  clip loss

                print("\nExtracting KD features...")
                KD_img_embeds = self.vision_model(KD_images)
                KD_text_embeds = self.text_model(KD_input_ids, KD_attention_mask)

                KD_logits_per_text = self.compute_logits(KD_img_embeds, KD_text_embeds)  #    Similarity
                KD_loss = self.softce_clip_loss(KD_logits_per_text, KD_target).to(self.device)  # CLIP clip loss

                #   Similarity is calculated using the attention mechanism
                KD_embed = self.attention(img_embeds.unsqueeze(0), KD_img_embeds.unsqueeze(0), KD_text_embeds.unsqueeze(0)).squeeze(0)

                mse_loss = torch.nn.MSELoss()
                KD_norm_loss = mse_loss(text_embeds, KD_embed)
                loss = loss + KD_loss + KD_norm_loss * 100

                loss = self.reduce_tensor(loss)                                 # loss   Multithreaded loss merge

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
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

        self.eval()
        return loss_ave / len(loader)

    #########################################
    #   prediction
    #########################################
    def forward(self, image, text):
        self.eval()
        #   pre process
        image = self.preprocess_image(image)
        text_input_ids, text_attention_mask = self.preprocess_text(text)

        with torch.no_grad():
            img_embeds = self.vision_model(image)
            text_embeds = self.text_model(text_input_ids, text_attention_mask)
            logits = self.compute_logits(img_embeds, text_embeds).t()
            probs = logits.softmax(dim=-1)

        return probs.cpu().numpy(), logits.cpu().numpy()

    #   img preprocess
    def preprocess_image(self, image):
        if image.dtype != np.float32:
            image = np.float32(image)

        #   norm
        if image.max() > 0:
            image /= 255
        if len(image.shape) > 2:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.expand_dims(image, 0)
        image = np.expand_dims(image, 0)
        #   Scale-invariant scaling
        image = torch.tensor(image)
        sizes = image.shape[-2:]
        max_size = max(sizes)
        scale = max_size / self.image_size
        image = torchvision.transforms.Resize((int(image.shape[-2] / scale), int(image.shape[-1] / scale)))(image)
        image = torch.nn.functional.pad(image, (0, self.image_size - image.shape[-1], 0, self.image_size - image.shape[-2], 0, 0))

        image = image.to(torch.float32).to(device)
        return image

    #   text preprocess
    def preprocess_text(self, text):
        prompts = [self.caption.replace("[CLS]", category) for category in text]
        text_tokens = self.text_model.tokenize(prompts)
        input_ids = text_tokens["input_ids"].to(device).to(torch.long)
        attention_mask = text_tokens["attention_mask"].to(device).to(torch.long)

        return input_ids, attention_mask

    #   text preprocess
    def compute_text_embeddings(self, categories, domain_knowledge=False):
        text_embeds_dict = {}                                                       #   Category name: text feature vector
        for iKey in range(len(categories)):
            if domain_knowledge and categories[iKey] in list(definitions.keys()):   # [+]  [Description + class name]
                descriptions = definitions[categories[iKey]]
                if categories[iKey] not in descriptions:
                    descriptions.append(categories[iKey])
            else:
                descriptions = [categories[iKey]]

            #   get text embed
            with torch.no_grad():
                print(descriptions)
                #   prompt template
                descriptions = [self.caption.replace("[CLS]", iDescription) for iDescription in descriptions]
                text_token = self.text_model.tokenizer(descriptions, truncation=True, padding=True, return_tensors='pt')
                input_ids = text_token["input_ids"].to(device).to(torch.long)
                attention_mask = text_token["attention_mask"].to(device).to(torch.long)
                #   get text feature
                text_embeds = self.text_model(input_ids, attention_mask)            #  *    Number of domain descriptions * feature dimension

            text_embeds_dict[categories[iKey]] = text_embeds.mean(0).unsqueeze(0)   # （）==》1 *    Average the text features of different descriptions

        text_embeds = torch.concat(list(text_embeds_dict.values()))

        return text_embeds_dict, text_embeds

