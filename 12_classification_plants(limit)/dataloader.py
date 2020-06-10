import torch
from torch.utils import data
from torchvision import datasets, transforms
import os
from PIL import Image

def get_transform(method=Image.BILINEAR):
    transform_list = []

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    
    return transforms.Compose(transform_list)


class CustomDataset(data.Dataset):
    def __init__(self, root, phase='train'):
        self.root = root
        self.phase = phase
        self.labels = {}
        
        self.label_path = os.path.join(root, self.phase, self.phase+'_labels.txt')
        with open(self.label_path, 'r', encoding='utf-8-sig') as f:
            file_list = []
            plant_list = []
            disease_list = []
            for line in f.readlines()[0:]:
                v = line.strip().split()
                file_list.append(v[0])
                if self.phase != 'test':
                    plant_list.append(v[1])
                    disease_list.append(v[2])

        self.labels['file'] = list(file_list)
        self.labels['plant'] = list(plant_list)
        self.labels['disease'] = list(disease_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.phase, self.labels['file'][index])
        
        if self.phase != 'test':
            plant = self.labels['plant'][index]
            plant = torch.tensor(int(plant))
            
            disease = self.labels['disease'][index]
            disease = torch.tensor(int(disease))

        transform = get_transform()
        image = Image.open(image_path).convert('RGB')
        image = transform(image)

        if self.phase != 'test' :
            return (self.labels['file'][index], image, plant, disease)
        elif self.phase == 'test' :
            dummy = ""
            return (self.labels['file'][index], image, dummy, dummy)

    def __len__(self):
        return len(self.labels['file'])

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
