import torch
from torch.utils import data
from torchvision import datasets, transforms
import os
from PIL import Image
import pandas as pd


class CustomDataset(data.Dataset):
    def __init__(self, root, phase='train'):
        self.root = root
        self.phase = phase
        self.labels = {}
        self.data_index_pool = [0, 1, 5, 6, 7, 19, 24, 25, 29, 31]
        self.label_index_pool = [2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 26, 27, 28, 30, 32, 33, 34]
        
        self.label_path = os.path.join(self.root, self.phase + '.csv')
        with open(self.label_path, 'r', encoding='utf-8-sig') as f:
            f.readline()
            f.readline()

            input_data = []
            output_data = []

            for line in f.readlines():
                values = line.strip().split(',')[2:]
                input_data.append([float(values[dindex]) for dindex in self.data_index_pool])
                output_data.append([float(values[lindex]) for lindex in self.label_index_pool])
        
        self.labels['input'] = input_data
        self.labels['output'] = output_data

    def __getitem__(self, index):
        input_data = torch.tensor(self.labels['input'][index])
        
        if self.phase != 'test' :
            output_data = torch.tensor(self.labels['output'][index])


        if self.phase != 'test' :
            return (input_data, output_data)
        elif self.phase == 'test' :
            dummy = []
            return (input_data,  dummy)

    def __len__(self):
        return len(self.labels['input'])

    def get_label_file(self):
        return self.label_path


def data_loader(root, phase='train', batch_size=16):
    if phase == 'train':
        shuffle = True
    else:
        shuffle = False

    dataset = CustomDataset(root, phase)
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset.get_label_file()
