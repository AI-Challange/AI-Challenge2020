import torch
from torch.utils import data
from torchvision import datasets, transforms
import os
from PIL import Image

def get_transform(method=Image.BILINEAR, normalize=True):
    transform_list = []

    transform_list.append(transforms.ToTensor())
    if normalize:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    return transforms.Compose(transform_list)


class CustomDataset(data.Dataset):
    def __init__(self, root, phase='train'):
        self.root = root
        self.phase = phase
        self.labels = {}
        
        self.label_path = os.path.join(root, self.phase, self.phase+'_labels.csv')
        with open(self.label_path, 'r', encoding='utf-8-sig') as f:
            input_list = []
            answer_list = []
            for line in f.readlines()[0:]:
                v = line.strip().split(',')
                input_list.append(v[0])
                if self.phase != 'test' :
                    answer_list.append(v[1])                

        self.labels['input'] = list(input_list)
        self.labels['answer'] = list(answer_list)

    def __getitem__(self, index):
        input_image_path = os.path.join(self.root, self.phase, self.labels['input'][index])

        transform_input = get_transform(normalize=True)
        input_image = Image.open(input_image_path).convert('RGB')
        input_image = transform_input(input_image)

        if self.phase != 'test' :
            answer_image_path = os.path.join(self.root, self.phase, self.labels['answer'][index])
            transform_answer = get_transform(normalize=False)
            answer_image = Image.open(answer_image_path).convert('RGB')
            answer_image = transform_answer(answer_image)

            return (self.labels['input'][index], input_image, self.labels['answer'][index], answer_image)
        elif self.phase == 'test' :
            dummy = ""
            return (self.labels['input'][index], input_image, dummy, dummy)

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