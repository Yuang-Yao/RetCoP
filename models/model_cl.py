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
from transformers import AutoModel, AutoTokenizer, logging           

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

PATH_PRETRAINED_WEIGHTS = "/mnt/data_ssd/yayao/CPT_retinal/RetCoP/checkpoint/"

#   Model initialization, architecture, training, inference
class RetCoPModel(torch.nn.Module):
    #########################################
    #   Model initialization module
    #########################################
    def __init__(self, vision_type='resnet_v1', bert_type='/mnt/data/yayao/CPT_data/emilyalsentzer/Bio_ClinicalBERT', vision_pretrained=True,
                 proj_dim=512, proj_bias=False, logit_scale_init_value=0.07, from_checkpoint=True, load_report=False, weights_path=None, weights_path_report=None,
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
        state_dict = torch.load(weights_path, map_location="cuda:0")
        self.load_state_dict(state_dict, strict=True)
        print('load model weight from:', weights_path)


    # /  Downstream task adaptation read/download pre-trained model parameters
    def load_from_pretrained_report(self, weights_path_report=None):
        if weights_path_report is None:
            raise ValueError("weights_path_report is None. Please provide a valid path.")
    
        print(f"Loading weights from: {weights_path_report}")

        device = f"cuda:{self.local_rank}"  #  local_rank 
        state_dict = torch.load(weights_path_report, map_location=device)
        # #   

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


        
        self.text_model.load_state_dict(filtered_state_dict, strict=False)
        print('load model weight from:', weights_path_report)


 
    

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

    #  text preprocess
    def preprocess_text(self, text):
        prompts = [self.caption.replace("[CLS]", category) for category in text]
        text_tokens = self.text_model.tokenize(prompts)
        input_ids = text_tokens["input_ids"].to(device).to(torch.long)
        attention_mask = text_tokens["attention_mask"].to(device).to(torch.long)

        return input_ids, attention_mask

 # text preprocess
    def preprocess_text0(self, text):
        #  text ，
        if isinstance(text, str):
            text = [text]
        
        #  text 
        if not text:
            raise ValueError("Input 'text' should be a non-empty list of strings.")

        #  tokenizer  text 
        text_tokens = self.text_model.tokenize(text)

        #  tokenizer  'input_ids'  'attention_mask'
        if "input_ids" not in text_tokens or "attention_mask" not in text_tokens:
            raise ValueError("Tokenization failed. Missing 'input_ids' or 'attention_mask' in tokenized output.")
        
        # 
        input_ids = text_tokens["input_ids"].to(self.device).to(torch.long)
        attention_mask = text_tokens["attention_mask"].to(self.device).to(torch.long)

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
                input_ids = text_token["input_ids"].to(self.device).to(torch.long)
                attention_mask = text_token["attention_mask"].to(self.device).to(torch.long)
                #   get text feature
                text_embeds = self.text_model(input_ids, attention_mask)            #      Number of domain descriptions * feature dimension

            text_embeds_dict[categories[iKey]] = text_embeds.mean(0).unsqueeze(0)   #  Average the text features of different descriptions

        text_embeds = torch.concat(list(text_embeds_dict.values()))

        return text_embeds_dict, text_embeds

