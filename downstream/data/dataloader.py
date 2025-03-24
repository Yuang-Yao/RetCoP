"""
data transformation (preprocessing); Split train, valid and test sets;
Create dataloader; Balance dataset categories
"""
                                                                                                               
import random
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from dataloader.dataset import Dataset
from dataloader.preprocessing import LoadImage, ImageScaling, CopyDict


# dataloader
# Data transformation (preprocessing); Split train, valid and test sets; Create dataloader
def get_dataloader_splits(dataframe_path, data_root_path, targets_dict, shots_train="80%", shots_val="0%",
                          shots_test="20%", balance=False, batch_size=8, num_workers=0, seed=0, task="classification",
                          size=(512, 512), batch_size_test=1, knowledge_dict=False, dataset=''):
    '''
    dataframe_path        data_root_path          targets_dict   category name
    seed=K 
    '''

    # （）   Set of data transformation (preprocessing) operations
    if task == "classification":
        transforms = Compose([CopyDict(), LoadImage(), ImageScaling(size=size)])
    else:
        transforms = Compose([CopyDict(), LoadImage(), ImageScaling()])

    #    Read the useful parts of the data dictionary (data & labels)
    #   Output: List of dictionaries
    data = []
    dataframe = pd.read_csv(dataframe_path)
    for i in range(len(dataframe)):
        sample_df = dataframe.loc[i, :].to_dict()                                      #   image,attributes,categories   Turn each line into a dictionary

        data_i = {"image_path": data_root_path + sample_df["image"]}                   #    Construct the required data
        if task == "classification":
            data_i["label"] = targets_dict[eval(sample_df["categories"])[0]]           # =》   Category Name = Number
            if dataset == 'Angiographic' and '%' in shots_train:
                # k-hot zslp
                khot_label = np.zeros(23, dtype=int)
                for cls in data_i["label"]:
                    khot_label[cls] = 1
                data_i["label"] = khot_label
            elif dataset == 'Angiographic' and '%' not in shots_train:
                #      fs
                if len(data_i["label"])>1:
                    continue
                else:
                    data_i["label"] = data_i["label"][0]

        data.append(data_i)

    # TAOP  TAOP test set
    if dataset == 'TAOP':
        data_test = []
        dataframe = pd.read_csv(dataframe_path.replace("train", "test"))
        for i in range(len(dataframe)):
            sample_df = dataframe.loc[i, :].to_dict()

            data_i = {"image_path": data_root_path + sample_df["image"]}
            if task == "classification":
                data_i["label"] = targets_dict[eval(sample_df["categories"])[0]]
            data_test.append(data_i)

    random.seed(seed)
    random.shuffle(data)

    #   Domain knowledge image text pair data
    if knowledge_dict:
        data_KD = []
        dataframe_KD = pd.read_csv("./local_data/dataframes/pretraining/39_MM_Retinal_dataset.csv")
        for i in range(len(dataframe_KD)):
            sample_df = dataframe_KD.loc[i, :].to_dict()
            data_i = {"image_path": data_root_path + sample_df["image"]}
            data_i["caption"] = sample_df["caption"]
            data_KD.append(data_i)

    # 、、       
    # Splitting the train, valid, and test sets
    # expected to have a consistent distribution of categories for each set
    labels = [data_i["label"] for data_i in data]                                       #   Label List

    if dataset == 'Angiographic' and '%' in shots_train:
        # 
        # zs 
        # fs10   
        # finetune  

        unique_labels = targets_dict.values()

        # unique_labelskhot
        unique_labels_khot = []
        for label in unique_labels:
            khot_label = np.zeros(23, dtype=int)
            for cls in label:
                khot_label[cls] = 1
            unique_labels_khot.append(khot_label)
        unique_labels = unique_labels_khot
    else:
        unique_labels = np.unique(labels)

    data_train, data_val, data_test = [], [], []

    for iLabel in unique_labels:
        if dataset == 'Angiographic':
            idx = list(np.squeeze([i for i, label in enumerate(labels) if np.all(label == iLabel)]))
        else:
            idx = list(np.squeeze(np.argwhere(labels == iLabel)))

        train_samples = get_shots(shots_train, len(idx))
        val_samples = get_shots(shots_val, len(idx))
        test_samples = get_shots(shots_test, len(idx))

        [data_test.append(data[iidx]) for iidx in idx[:test_samples]]
        [data_train.append(data[iidx]) for iidx in idx[test_samples:test_samples+train_samples]]
        [data_val.append(data[iidx]) for iidx in idx[test_samples+train_samples:test_samples+train_samples+val_samples]]

    if balance:
        data_train = balance_data(data_train)

    if dataset == 'TAOP':
        train_loader = get_loader(data, transforms, "train", batch_size, num_workers)
        val_loader = None
        test_loader = get_loader(data_test, transforms, "test", batch_size_test, num_workers)
    else:
        train_loader = get_loader(data_train, transforms, "train", batch_size, num_workers)
        val_loader = get_loader(data_val, transforms, "val", batch_size_test, num_workers)
        test_loader = get_loader(data_test, transforms, "test", batch_size_test, num_workers)

    KD_loader = None
    if knowledge_dict:
        KD_loader = get_loader(data_KD, transforms, "KD", batch_size, num_workers)

    loaders = {"train": train_loader, "val": val_loader, "test": test_loader, "KD":KD_loader}
    return loaders


# dataset + dataloader
def get_loader(data, transforms, split, batch_size, num_workers):

    if len(data) == 0:
        loader = None
    else:
        dataset = Dataset(data=data, transform=transforms)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle = split == "train", num_workers=num_workers, drop_last=False, pin_memory=False)
    return loader


#   Balance the distribution of different categories in the training set
def balance_data(data):
    labels = [iSample["label"] for iSample in data]                     # label  label of all sample
    unique_labels = np.unique(labels)                                   
    counts = np.bincount(labels)                                        #   Count the number of samples for each category

    N_max = np.max(counts)

    data_out = []
    for iLabel in unique_labels:
        idx = list(np.argwhere(np.array(labels) == iLabel)[:, 0])

        # If the current category is less in the training set, some samples are randomly selected to be repeated,
        # so that the number of all categories is the same
        if N_max-counts[iLabel] > 0:
            idx += random.choices(idx, k=N_max-counts[iLabel])
        [data_out.append(data[iidx]) for iidx in idx]

    return data_out


#   Return sample number
def get_shots(shots_str, N):
    #   Input percentage
    if "%" in str(shots_str):
        shots_int = int(int(shots_str[:-1]) / 100 * N)
    #   Direct input number
    else:
        shots_int = int(shots_str)
    return shots_int
