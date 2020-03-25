import torch
import numpy as np
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import os
import pandas as pd
from PIL import Image
import random


def load_pict(load_path, transform=None):
    class MyDataset(Dataset):
        def __init__(self, file_path, transform):
            self.df = pd.read_csv(file_path)
            self.transform = transform

        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            img_path = self.df.iloc[idx]["image_path"]
            label = self.df.iloc[idx]["label"]
            img = Image.open(img_path)
            if self.transform:
                img = self.transform(img)
            return img, label
    return MyDataset(load_path, transform=transform)

def concat_dataset(s_trainset, t_trainset):
    class catDataset(Dataset):
        def __init__(self, s_trainset, t_trainset, transform = None):
            self.ds1 = s_trainset
            self.ds2 = t_trainset
            self.transform = transform
        def __len__(self):
            return max(len(self.ds1), len(self.ds2))
        def __getitem__(self, idx):
            sample = {
                "s_data" : self.ds1[idx % len(self.ds1)][0],
                "s_label" : self.ds1[idx % len(self.ds1)][1],
                "t_data" : self.ds2[idx % len(self.ds2)][0],
                "t_label" : self.ds2[idx % len(self.ds2)][1]
            }
            if self.transform:
                sample["s_data"] = self.transform(sample["s_data"])
                sample["t_data"] = self.transform(sample["t_data"])
            
            return sample

    return catDataset(s_trainset, t_trainset)

def dataset_sampler(dataset, class_num=10, data_per_class=10):
    datas = []
    labels = []
    for cln in range(class_num):
        cand = []
        lbl = []
        cnt = 0
        for sample in dataset:
            data, label = sample
            if cln == int(label):
                cand.append(data)
                lbl.append(cln)
                cnt+=1
            if cnt == data_per_class:
                break
        datas+=cand
        labels+=lbl
    print("number of sampled target dataset: {}".format(len(labels)))

    class MyDataset(Dataset):
        def __init__(self, data, label):
            self.data = data
            self.label = label
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            img = self.data[idx]
            label = self.label[idx]
            return img, label
    
    return MyDataset(datas, labels)

