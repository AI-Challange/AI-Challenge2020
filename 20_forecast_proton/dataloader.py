#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from glob import glob
import csv
import collections
import os
import pandas as pd

class CustomDataset(data.Dataset):

    def __init__(self, root, phase='train', transform=None):
        
        self.root = os.path.join(root, phase)
        csv_paths = sorted(glob(os.path.join(self.root, '*.csv')))
        input_paths = [csv_paths[0], csv_paths[1], csv_paths[3]]
        output_path = csv_paths[2]
        pd_labels = pd.read_csv(output_path)
        self.labels = np.array(pd_labels)
        self.input = np.array([[] for i in range(len(self.labels))])
        for input_path in input_paths:
            self.input = np.append(self.input, group_time(input_path, self.labels), axis=1)
        
        self.transform = transform
        

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        inp = self.input[index]
        label = self.labels[index]
        # input data 중 minus 값을 0으로 변환
        inp = [0 if val < 0 else val for val in inp]
        inp = torch.as_tensor(inp, dtype=torch.float)
        label = torch.tensor(label[1])
        if self.transform is not None:
            inp, label = self.transform(inp, label)
        return (inp, label)

def data_loader(root, batch_size, phase='train', transform=None) :
    dataset = CustomDataset(root, phase=phase, transform=transform)
    shuffle = False
    if phase == 'train' :
        shuffle = True
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def group_time(csv_path, label) :
    print(csv_path)
    csv_df = pd.read_csv(csv_path)
    data = np.array(csv_df)
    data_it = iter(data) #np.array
    default_val = ['-1:-1'] + [-1 for i in range(data.shape[1]-1)]
    each_data = next(data_it, default_val)
    data_time = float(each_data[0].split(':')[1])
    data_item = np.array(each_data[1:], dtype=np.float64)
    new_datas = []
    times = []
    cal_default = [0.0 for i in range(data.shape[1]-1)]
    mean_data = np.array(cal_default)

    for idx, val in enumerate(label) :
        total_data = np.array(cal_default)
        time = val[0]
        times.append(time)
        minute = float(time.split(':')[1])
        data_count = 0
        while minute <= data_time < minute + 5 :
            total_data += data_item
            data_count += 1
            each_data = next(data_it, default_val)
            data_time = float(each_data[0].split(':')[1])
            data_item = np.array(each_data[1:], dtype=np.float64)
        if data_count != 0 :
            mean_data = total_data / data_count
        new_datas.append(mean_data)
    print(csv_path, ' end')
    return np.array(new_datas)
