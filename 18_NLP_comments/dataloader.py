import torch
from torch.utils import data
from torchvision import datasets, transforms
import os
from PIL import Image
import pandas as pd


class CustomDataset(data.Dataset):
    def __init__(self, root, phase='train'):
        count_vectorizer = make_vocab(root)
        self.root = root
        self.phase = phase
        self.labels = {}

        self.label_path = os.path.join(root, self.phase + '_hate.txt')
        with open(self.label_path, 'r') as f:
            comments_list = []
            bias_list = []
            hate_list = []
            for line in f.readlines()[0:]:
                v = line.strip().split('\t')
                comments_list.append(v[0] + v[1])
                if phase != 'test':
                    bias_list.append(v[2])
                    hate_list.append(v[3])

        comments_vector = []
        for comment in comments_list:
            comments_vector.append(count_vectorizer.transform([comment]).toarray()[0])
        comments_vector = torch.FloatTensor(comments_vector)

        self.comments_vec = comments_vector  # 문장 벡터
        self.comments_list = comments_list  # 문장 원본
        if self.phase != 'test':
            bias_name_list = ['none', 'gender', 'others']
            hate_name_list = ['none', 'hate', 'offensive']
            from itertools import product
            bias_hate_list = [bias_name_list, hate_name_list]
            bias_hate_list = list(product(*bias_hate_list))
            label_list = []
            for idx in range(len(comments_list)):
                labels = (bias_list[idx], hate_list[idx])
                label_list.append(bias_hate_list.index(labels))
            self.label_list = label_list

    def __getitem__(self, index):
        if self.phase != 'test':
            return (self.comments_list[index], self.comments_vec[index]), self.label_list[index]
        elif self.phase == 'test':
            dummy = ""
            return (self.comments_list[index], self.comments_vec[index]), dummy

    def __len__(self):
        return len(self.comments_list)


def make_vocab(root):
    comments_list = []
    phases = ['train', 'test', 'validate']
    for phase in phases:
        with open(root + '/' + phase + '_hate.txt', 'r') as f:
            for line in f.readlines()[0:]:
                v = line.strip().split('\t')
                comments_list.append(v[0] + v[1])


    from sklearn.feature_extraction.text import CountVectorizer
    count_vectorizer = CountVectorizer()
    count_vectorizer.fit(comments_list)
    return count_vectorizer


def data_loader(root, phase='train', batch_size=16):
    if phase == 'train':
        shuffle = True
    else:
        shuffle = False

    dataset = CustomDataset(root, phase)
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

