import os
import numpy as np
import torch
from PIL import Image
from glob import glob
import xml.etree.ElementTree as elemTree
import random
import cv2
from torchvision.transforms import functional as F

class CustomDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # 모든 이미지 파일들을 읽고, 정렬하여
        # 이미지와 분할 마스크 정렬을 확인합니다
        self.imgs = list(sorted(glob(os.path.join(root, '*.jpg'))))
        self.labels = elemTree.parse(glob(root + '/*.xml')[0])
        #self.masks = list(sorted(os.listdir(os.path.join(root, 'MASK'))))

    def __getitem__(self, idx):
        # 이미지와 마스크를 읽어옵니다
        img_path = self.imgs[idx]
        #print(img_path)

        #mask_path = os.path.join(self.root, 'MASK', self.masks[idx])
        img = Image.open(img_path).convert('RGB')
        h, w = img.size
        # 분할 마스크는 RGB로 변환하지 않음을 유의하세요
        # 왜냐하면 각 색상은 다른 인스턴스에 해당하며, 0은 배경에 해당합니다
        label = self.labels.findall('./image')[idx]

        masks = []
        boxes = []
        class_names = []
        class_num = {'sidewalk_blocks' : 1, 'alley_damaged' : 2, 'sidewalk_damaged' : 3, 'caution_zone_manhole': 4, 'braille_guide_blocks_damaged':5,\
                    'alley_speed_bump':6,'roadway_crosswalk':7,'sidewalk_urethane':8, 'caution_zone_repair_zone':9, 'sidewalk_asphalt':10, 'sidewalk_other':11,\
                    'alley_crosswalk':12,'caution_zone_tree_zone':13, 'caution_zone_grating':14, 'roadway_normal':15, 'bike_lane':16, 'caution_zone_stairs':17,\
                    'alley_normal':18, 'sidewalk_cement':19,'braille_guide_blocks_normal':20, 'sidewalk_soil_stone': 21}
        for polygon_info in label.findall('./polygon') :
            class_name,x1,y1,x2,y2 = None, None, None, None, None
            class_name,points = polygon_info.attrib['label'], polygon_info.attrib['points']
            points = points.split(';')
            pos = []
            for p in points:
                x , y = p.split(',')
                pos.append([int(float(x)), int(float(y))])
            temp = []
            pos  = np.array(pos)
            #print(pos.shape)
            if class_name != 'bike_lane' :
                if len(polygon_info.findall('attribute')) == 0:
                    continue#print(polygon_info.findall('attribute')[0].text)
                class_name += '_'+polygon_info.findall('attribute')[0].text
            x1 = np.min(pos[:, 0])
            y1 = np.min(pos[:, 1])
            x2 = np.max(pos[:, 0])
            y2 = np.max([pos[:, 1]])
            boxes.append([x1,y1,x2,y2])
            class_names.append(class_num[class_name])
            canvas = np.zeros((w, h), np.uint8)
            cv2.fillPoly(canvas, [pos], (255))
            mask = canvas == 255
            masks.append(mask)
        # 모든 것을 torch.Tensor 타입으로 변환합니다
        #print(len(masks), len(class_names))
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # 객체 종류는 한 종류만 존재합니다(역자주: 예제에서는 사람만이 대상입니다) NO!!
        class_names = torch.as_tensor(class_names, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        #area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # 모든 인스턴스는 군중(crowd) 상태가 아님을 가정합니다
        # iscrowd = torch.zeros((len(class_names),), dtype=torch.int64)
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = class_names
        target["masks"] = masks
        target["image_id"] = image_id
        # target["area"] = area
        # target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target

    def __len__(self):
        return len(self.imgs)

class Testloader(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # 모든 이미지 파일들을 읽고, 정렬하여
        # 이미지와 분할 마스크 정렬을 확인합니다
        self.imgs = list(sorted(glob(os.path.join(root, '*.jpg'))))


    def __getitem__(self, idx):
        # 이미지와 마스크를 읽어옵니다
        img_path = self.imgs[idx]
        img_name = img_path.split('/')[-1].split('.')[0]

        img = Image.open(img_path).convert('RGB')
        h, w = img.size
        
        target = {}
        target["image_id"] = img_name
     

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target

    def __len__(self):
        return len(self.imgs)
    
    

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
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
    

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        # (역자주: 학습시 50% 확률로 학습 영상을 좌우 반전 변환합니다)
        transforms.append(T.RandomHorizontalFlip(0.3))
    return T.Compose(transforms)
                   
def collate_fn(batch):
    return tuple(zip(*batch))
                   
def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    if train:
        # (역자주: 학습시 50% 확률로 학습 영상을 좌우 반전 변환합니다)
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)
                   
def collate_fn(batch):
    return tuple(zip(*batch))

def make_dataset(root):
    folder_list = glob(root+'/*')
    dataset = CustomDataset(folder_list[0], get_transform(train=False))
    for fpath in folder_list[1:]:
        dataset_test = CustomDataset(fpath, get_transform(train=False))
        dataset = torch.utils.data.ConcatDataset([dataset, dataset_test])
    return dataset

def make_testset(root):
    folder_list = glob(root+'/*')
    dataset = Testloader(folder_list[0], get_transform(train=False))
    for fpath in folder_list[1:]:
        dataset_test = Testloader(fpath, get_transform(train=False))
        dataset = torch.utils.data.ConcatDataset([dataset, dataset_test])
    return dataset