import os
import pickle
import re
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
from transformers import BertTokenizer
from transformers import AutoTokenizer
import cv2
import pydicom
from PIL import Image
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizerFast,
    RobertaTokenizerFast
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))



class get_loader_report(data.Dataset):
    def __init__(self, data_path, split="train", transform=None, data_pct=1.0,
                 imsize=256, max_words=112, sent_num=3):
        super().__init__()
        if not os.path.exists(data_path):
            raise RuntimeError(f"{data_path} does not exist!")

        self.transform = transform
        self.imsize = imsize
        self.data_path = data_path
        self.split = split
        self.data_pct = data_pct
        self.max_words = max_words

        # Load text data from all txt files in the directory
        self.filenames, self.path2sent = self.load_text_data()

        # Initialize tokenizer and collator
        model_path = "/mnt/data/yayao/CPT_data/emilyalsentzer/Bio_ClinicalBERT"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.mlm_collator = DataCollatorForWholeWordMask(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)

        print("report sample number:", len(self.filenames))

    def load_text_data(self):
        # Get all txt files in the directory
        txt_files = [f for f in os.listdir(self.data_path) if f.endswith('.txt')]

        # Filter files based on split and data percentage
        if self.split == "train" and self.data_pct != 1.0:
            txt_files = random.sample(txt_files, int(len(txt_files) * self.data_pct))

        filenames = []
        path2sent = {}

        for txt_file in txt_files:
            file_path = os.path.join(self.data_path, txt_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                paragraphs = content.split('\n\n')  # Assuming paragraphs are separated by double newlines
                for paragraph in paragraphs:
                    if paragraph.strip():  # Ensure the paragraph is not empty
                        path2sent[file_path + '_' + str(len(filenames))] = paragraph.strip()
                        filenames.append(file_path + '_' + str(len(filenames)))

        return filenames, path2sent

    def create_path_2_sent_mapping(self):
        sent_lens, num_sents = [], []
        path2sent = {}

        for path, content in tqdm(self.path2sent.items(), total=len(self.path2sent)):
            # Process the content as needed
            captions = content.replace("\n", " ")

            # Split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # Create tokens from captions
            for cap in captions:
                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                if len(tokens) <= 1:
                    continue

                # Filter tokens for current sentence
                included_tokens = [t.encode("ascii", "ignore").decode("ascii") for t in tokens if len(t) > 0]

                if len(included_tokens) > 0:
                    study_sent.append(" ".join(included_tokens))

                cnt += len(included_tokens)

            if cnt >= 3:
                sent_lens.append(cnt)
                num_sents.append(len(study_sent))
                path2sent[path] = study_sent

        # Get report word/sentence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)

        print(f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]")
        print(f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]")

        return path2sent

    def __len__(self):
        return len(self.filenames)

    def get_caption(self, path):
        content = self.path2sent[path]

        if not content:
            raise Exception("no content for path")

        # Separate different sentences
        sentences = list(filter(lambda x: x != "", content.split("\n")))
        sent = " ".join(sentences)

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def __getitem__(self, index):
        key = self.filenames[index]
        caps, cap_len = self.get_caption(key)
        text, attention_mask = caps["input_ids"], caps["attention_mask"]
        caps_mask = self.mlm_collator(tuple(text))

        return caps_mask["input_ids"].squeeze(0), caps_mask["labels"].squeeze(0), attention_mask.squeeze(0), text.squeeze(0)


