"""
Dataset and Dataloader   Dataset and Dataloader main function
"""
import random

import numpy as np
import pandas as pd
import torch
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from dataloader.dataset import Dataset, UniformDataset
from dataloader.preprocessing import LoadImage, ImageScaling, SelectRelevantKeys, CopyDict,\
    ProduceDescription, AugmentDescription, LoadTensor


def get_loader(dataframes_path, data_root_path, datasets, balance=False, batch_size=8, num_workers=0,
               banned_categories=None, caption="A fundus photograph of [CLS]", augment_description=True,
               knowledge_dict=True, modality = "CFP", classification=0, test_code=False, feature_pool=False,
               fusion=False,device="cuda"):
    """
    datasets list
    banned_categories list
    """
    Label_dict={"Macular diseases":0, "Retinal vascular diseases":1, "Retinal and choroidal inflammatory diseases":2,
                "Chorio-retinal degeneration and dystrophy":3, "Intraocular tumors":4, "Disease of congenital anomalies":5,
                "Optic nerve diseases":6, "Traumatic and toxic retinopathies":7, "Normal":8, "Disease of vitreous":9,
                "Retinal and choroidal detachment":10, "Fundus disease caused by systemic disease":11,
                "Anterior segment diseases":12}

    #   Set of data transformation (preprocessing) operations
    if modality == "CFP":
        transforms = Compose([
            CopyDict(),                                             #     Deep copy data. Data is stored in dictionary
            LoadImage(),                                            # 、、  Read pictures, preprocess, and store picture tensors into dictionaries
            # ImageScaling(),                                       #   Image resize to model input size [unchanged aspect ratio]
            # LoadTensor(),
            ProduceDescription(caption=caption),                    # text  Generate raw text
            AugmentDescription(augment=augment_description),        #   change the class name into medical description
            SelectRelevantKeys(classification=classification)                                    #   Returns the data used for training
        ])
    elif modality == "FFA":
        transforms = Compose([
            CopyDict(),
            LoadImage(),
            ProduceDescription(caption=caption),
            SelectRelevantKeys(classification=classification)
        ])
    elif modality == "OCT":
        transforms = Compose([
            CopyDict(),
            LoadImage(),
            ProduceDescription(caption=caption),
            SelectRelevantKeys(classification=classification)
        ])
    if knowledge_dict:
        KD_transforms = Compose([CopyDict(), LoadImage()])      # MM  MM dataset

    #   assembly all the dataset
    print("Setting assembly data...")
    data = []
    for iDataset in datasets:
        print("Processing data: " + iDataset)

        dataframe = pd.read_csv(dataframes_path + iDataset + ".csv", low_memory=False)
        dataframe.dropna(how='all', inplace=True)
        dataframe = dataframe.applymap(lambda x: "" if pd.isna(x) else x)

        selected_id_list = range(len(dataframe))                      # 100%   100% data

        for i in selected_id_list:
            data_i = dataframe.loc[i, :].to_dict()              #   image,attributes,categories   Turn each line into a dictionary
            data_i["categories"] = eval(data_i["categories"])
            data_i["atributes"] = eval(data_i["atributes"])
            if classification:
                Label_str = data_i["Label"]
                if Label_str=="":
                    data_i["Label"]=np.zeros(classification, dtype=int)
                else:
                    Label_str_list = Label_str.split(", ")
                    data_i["Label"] = np.zeros(classification, dtype=int)
                    for j in range(len(Label_str_list)):
                        data_i["Label"][Label_dict[Label_str_list[j].strip().replace("\n", " ")]] = 1

            # ban   Removes the banned category
            banned = False
            if banned_categories is not None:
                for iCat in data_i["categories"]:
                    if iCat in banned_categories:
                        banned = True
            if banned:
                continue

            # 
            if fusion and data_i["Label"].sum()==0:
                continue

            # ban    The total data set that is not banned
            data_i["image_name"] = data_i["image"]
            if modality == "CFP":
                data_i["image_path"] = data_root_path + data_i["image"]
            elif modality == "FFA":
                data_i["image_path"] = "/mnt/data/jlzhang/Dataset/Resized/" + data_i["image"]
            elif modality == "OCT":
                data_i["image_path"] = data_root_path + data_i["image"]
            data.append(data_i)
    print('Total assembly data samples: {}'.format(len(data)))

    #    Domain knowledge image text pair data
    if knowledge_dict:
        # MM  MM dataset
        data_KD = []
        if modality=='CFP':
            dataframe_KD = pd.read_csv("/mnt/data_ssd/yayao/CPT_retinal/RetCoP/dataframes/CFP/pretraining/39_MM_Retinal_dataset.csv")
        elif modality=='FFA':
            dataframe_KD = pd.read_csv("/mnt/data_ssd/yayao/CPT_retinal/RetCoP/dataframes/FFA/pretraining/MM-Retinal_FFA.csv")
        elif modality=='OCT':
            dataframe_KD = pd.read_csv("/mnt/data_ssd/yayao/CPT_retinal/RetCoP/dataframes/OCT/pretraining/OCT17_MM-Retinal_OCT.csv")

        dataframe_KD = dataframe_KD.applymap(lambda x: '' if pd.isna(x) else x)
        for i in range(len(dataframe_KD)):
            sample_df = dataframe_KD.loc[i, :].to_dict()
            data_i = {"image_path": data_root_path + sample_df["Image_ID"]}
            data_i["caption"] = sample_df["en_caption"]
            if classification:
                Label_str = sample_df["Label"]
                if Label_str == "":
                    data_i["Label"] = np.zeros(classification, dtype=int)
                else:
                    Label_str_list = Label_str.split(", ")
                    data_i["Label"] = np.zeros(classification, dtype=int)
                    for j in range(len(Label_str_list)):
                        data_i["Label"][Label_dict[Label_str_list[j].strip().replace("\n", " ")]] = 1

            # 
            if fusion and data_i["Label"].sum()==0:
                continue

            data_KD.append(data_i)



    #    train set
    if balance:
        train_dataset = UniformDataset(data=data, transform=transforms)
    else:
        train_dataset = Dataset(data=data, transform=transforms)

    #    knowledge dictionary
    if knowledge_dict:
        KD_dataset = Dataset(data=data_KD, transform=KD_transforms)


    # data sampler
    if not test_code:
        train_sampler = DistributedSampler(train_dataset)                   #   distributed training
        if fusion:
            #  KD
            KD_train_sampler = DistributedSampler(KD_dataset)

    
    # dataloader
    KD_loader = None
    if test_code:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)         #   1 gpu
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)  #   mutil gpus

    if knowledge_dict:
        if feature_pool or test_code:
            #  KD
            KD_loader = DataLoader(KD_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        elif fusion and not test_code:
            # 
            KD_loader = DataLoader(KD_dataset, batch_size=batch_size, num_workers=num_workers, sampler=KD_train_sampler)
        else:
            # KD/MM     【】   Training KD/MM with training data requires adjustments to the data extraction
            weights = torch.ones(len(KD_dataset))  #   uniform distribution
            weightedRandomSampler = WeightedRandomSampler(weights=weights, replacement=True, num_samples=batch_size * len(train_loader))    #   The amount sampled is consistent with the training set
            KD_loader = DataLoader(KD_dataset, batch_size=batch_size, num_workers=num_workers, sampler=weightedRandomSampler)
  
    dataloaders = {"train": train_loader, "KD":KD_loader}
    return dataloaders




#  get_loader 
def get_loader_cus1(batch_size=8, num_workers=0,test_code=False,device="cuda", dataset=None):


    # ，
    if dataset is not None:
        train_dataset = dataset

    # data sampler
    if not test_code:
        train_sampler = DistributedSampler(train_dataset)                   #   distributed training
    else:
        train_sampler = None

    if test_code:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)         #   1 gpu
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)  #   mutil gpus

 
    return train_loader



#  get_loader 
def get_loader_cus2( batch_size=8, num_workers=0,device="cuda", dataset=None,train_loader=None):


    # ，
    if dataset is not None:
        KD_dataset = dataset

    # dataloader
    KD_loader = None

    # KD/MM      Training KD/MM with training data requires adjustments to the data extraction
    weights = torch.ones(len(KD_dataset))  #   uniform distribution
    weightedRandomSampler = WeightedRandomSampler(weights=weights, replacement=True, num_samples=batch_size * len(train_loader))    #   The amount sampled is consistent with the training set
    KD_loader = DataLoader(KD_dataset, batch_size=batch_size, num_workers=num_workers, sampler=weightedRandomSampler)

    return KD_loader