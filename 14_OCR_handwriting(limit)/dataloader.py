#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from glob import glob
import json
import collections
import os

class CustomDataset(data.Dataset):

    def __init__(self, root, phase='train', transform=None, target_transform=None):
        
        self.root = os.path.join(root, phase)
        self.imgs = sorted(glob(self.root + '/*.png')+glob(self.root + '/*.jpg'))
        self.labels = []
        self.transform = transform
        self.target_transform = target_transform
        annotations = None
    
        with open(os.path.join(self.root, phase + '.json'), 'r') as label_json :
            label_json = json.load(label_json)
            annotations = label_json['annotations']
        annotations = sorted(annotations, key=lambda x: x['file_name'])
        for anno in annotations :
            if phase == 'test' :
                self.labels.append('dummy')
            else :
                self.labels.append(anno['text'])
    


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.imgs[index]
        img = Image.open(img_path).convert('L')
        label = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, label)
    def get_root(self) :
        return self.root

    def get_img_path(self, index) :
        return self.imgs[index]


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class alignCollate(object):

    def __init__(self, imgH=32, imgW=100):
        self.imgH = imgH
        self.imgW = imgW

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels


class strLabelConverter(object):
    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts



def loadData(v, data):
    d_size = data.size()
    v.resize_(d_size).copy_(data)

def data_loader(root, batch_size, imgH, imgW, phase='train', transform=None, target_transform=None) :
    dataset = CustomDataset(root=root, phase=phase, transform=transform, target_transform=target_transform)
    shuffle = False
    if phase == 'train' :
        shuffle = True
    dataloader = data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=shuffle, collate_fn=alignCollate(imgH=imgH, imgW=imgW))
    
    return dataloader
