# -*- coding: UTF-8 -*-
import torch
from torch.utils import data
from torchvision import datasets, transforms
import os
import codecs
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
def get_transform(resize=112, method=Image.BILINEAR):
    transform_list = []
    if resize > 0:
        size = [resize, resize]
        transform_list.append(transforms.Resize(size, method))

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(transform_list)

class CustomDataset(data.Dataset):
    def __init__(self, root, phase='train', max_vector= 100):
        self.root = root
        self.phase = phase
        self.max_vector = max_vector
        self.labels = {}
        

        self.label_path = os.path.join(root, self.phase, self.phase+'_label.txt')
        with codecs.open(self.label_path, 'r', encoding='utf8') as f:
            file_list = []
            label_list = []
            for line in f.readlines()[0:]:
                v = line.strip().split()
                file_list.append(v[0])
                
                if self.phase != 'test' :
                    label = v[1:]
                    label_list.append(label)      
        self.labels['file'] = list(file_list)
        self.labels['label'] = list(label_list)

    def __getitem__(self, index):
        sound_path = os.path.join(self.root, self.phase, self.labels['file'][index])
        
        if self.phase != 'test' :
            label = self.labels['label'][index]
            ascii_ = [x.encode('utf8') for x in label]
            label = torch.LongTensor(ascii_)
            tmp = torch.zeros(1,self.max_vector)
            for i in range(len(label[0])) :
                tmp[0][i] = label[0][i]
            is_label = tmp
            is_label = is_label.squeeze_()
        
        with open( sound_path, 'rb') as opened_pcm_file:
            buf = opened_pcm_file.read()
            pcm_data = np.frombuffer(buf, dtype = 'int16')
            tmp = np.zeros((1,200000))
            for i in range(len(pcm_data)) :
                tmp[0][i] = pcm_data[i]
            tmp = torch.from_numpy(tmp).float()
            tmp = tmp.squeeze_()
        
        if self.phase != 'test' :
            return (self.labels['file'][index], tmp, is_label)
        elif self.phase == 'test' :
            dummy = ""
            return (self.labels['file'][index], tmp, dummy)


    def __len__(self):
        return len(self.labels['file'])

    def get_label_file(self):
        return self.label_path


def data_loader(root, phase='train', batch_size=16, max_vector= 100):
    if phase == 'train':
        shuffle = True
    else:
        shuffle = False

    dataset = CustomDataset(root, phase, max_vector)
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset.get_label_file()