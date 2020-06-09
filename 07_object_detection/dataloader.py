import os
import numpy as np
import torch
from PIL import Image
from glob import glob
import xml.etree.ElementTree as elemTree
import random
from torchvision.transforms import functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class CustomDataset(object):
    def __init__(self, root, transforms, phase='train'):
        self.root = os.path.join(root, phase)
        self.transforms = transforms
        self.imgs = sorted(list(glob(self.root + '/*jpg') + list(glob(self.root + '/*png'))))
        self.labels = sorted(elemTree.parse(os.path.join(self.root, phase + '.xml')).findall('image'), key=lambda x: x.attrib['name'])

    def __getitem__(self, idx):
        # 이미지와 마스크를 읽어옵니다
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        boxes = []
        class_names = []
        class_num = {'bus' : 1, 'car' : 2, 'carrier' : 3, 'cat' : 4, 'dog' : 5, 'motorcycle' : 6, 'movable_signage' : 7,
                     'person' : 8, 'scooter' : 9, 'stroller' : 10, 'truck' : 11, 'wheelchair' : 12, 'barricade' : 13, 'bench' : 14, 'chair' : 15,
                     'fire_hydrant' : 16, 'kiosk': 17, 'parking_meter' : 18, 'pole': 19, 'potted_plant' : 20, 'power_controller' : 21, 'stop' : 22, 'table' : 23,
                     'traffic_light_controller':24, 'traffic_sign':25, 'tree_trunk':26, 'bollard':27, 'bicycle' : 28}
                     
        for box_info in label.findall('./box') :
            class_name,x1,y1,x2,y2 = box_info.attrib['label'],box_info.attrib['xtl'],box_info.attrib['ytl'],box_info.attrib['xbr'],box_info.attrib['ybr'] 
            x1,y1,x2,y2 = map(int,map(float, [x1,y1,x2,y2]))
            boxes.append([x1,y1,x2,y2])
            class_names.append(class_num[class_name])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        class_names = torch.as_tensor(class_names, dtype=torch.int64)

        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = boxes
        target["labels"] = class_names
        target["image_id"] = image_id


        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target

    def __len__(self):
        return len(self.imgs)

    def get_img_path(self, index) :
        return self.imgs[index]

def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)
                   
def collate_fn(batch):
    return tuple(zip(*batch))
                   
def data_loader(root, batch_size, phase='train') :
    is_train = False
    shuffle = False
    if phase == 'train' :
        is_train = True
        shuffle = True
    dataset = CustomDataset(root, get_transform(train=is_train), phase=phase)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    return dataloader
