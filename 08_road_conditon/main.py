import dataloader
from dataloader import CustomDataset
import model
import torch
import torch.distributed as dist
import math
import argparse
import os
import sys
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/.../path/anaconda3/lib/
from imantics import Polygons, Mask
from shapely.geometry import Polygon
import xml.etree.ElementTree as elemTree
import numpy as np
import sys

'''
!!!!!!!!!!!!!!!!!!!!! 필독!!!!!!!!!!!!!!!!!!!!!!!!!!!
** 컨테이너 내 기본 제공 폴더
- /datasets : read only 폴더 (각 태스크를 위한 데이터셋 제공)
- /tf/notebooks :  read/write 폴더 (참가자가 Wirte 용도로 사용할 폴더)
1. 참가자는 /datasets 폴더에 주어진 데이터셋을 적절한 폴더(/tf/notebooks) 내에 복사/압축해제 등을 진행한 뒤 사용해야합니다.
   예시> Jpyter Notebook 환경에서 압축 해제 예시 : !bash -c "unzip /datasets/objstrgzip/18_NLP_comments.zip -d /tf/notebooks/
   예시> Terminal(Vs Code) 환경에서 압축 해제 예시 : bash -c "unzip /datasets/objstrgzip/18_NLP_comments.zip -d /tf/notebooks/
   
2. 참가자는 각 문제별로 데이터를 로드하기 위해 적절한 path를 코드에 입력해야합니다. (main.py 참조)
3. 참가자는 모델의 결과 파일(Ex> prediction.txt)을 write가 가능한 폴더에 저장되도록 적절 한 path를 입력해야합니다. (main.py 참조)
4. 세션/컨테이너 등 재시작시 위에 명시된 폴더(datasets, notebooks) 외에는 삭제될 수 있으니 
   참가자는 적절한 폴더에 Dataset, Source code, 결과 파일 등을 저장한 뒤 활용해야합니다.
   
!!!!!!!!!!!!!!!!!!!!! 필독!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''

try:
    from nipa import nipa_data
    DATASET_PATH = nipa_data.get_data_root('deepfake')
except:
    DATASET_PATH = os.path.join('./data/08_road_conditon/test')

def save_model(model_name, model, optimizer, scheduler):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(state, os.path.join(model_name + '.pth'))
    print('model saved')
    
def load_model(model_name, model, optimizer=None, scheduler=None):
    state = torch.load(os.path.join(model_name))
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    print('model loaded')
    
def test(model, data_loader_test, device, predict_path) :
    class_name = {1 :'sidewalk_blocks' , 2 : 'alley_damaged', 3 :'sidewalk_damaged', 4 : 'caution_zone_manhole', 5 : 'braille_guide_blocks_damaged', 
          6 : 'alley_speed_bump', 7 : 'roadway_crosswalk', 8 : 'sidewalk_urethane', 9 : 'caution_zone_repair_zone', 10 : 'sidewalk_asphalt', 
          11 : 'sidewalk_other', 12 : 'alley_crosswalk', 13 : 'caution_zone_tree_zone', 14 : 'caution_zone_grating', 15 : 'roadway_normal',
          16 : 'bike_lane', 17 : 'caution_zone_stairs', 18 : 'alley_normal', 19 : 'sidewalk_cement', 20 : 'braille_guide_blocks_normal', 
          21 : 'sidewalk_soil_stone'}
   
    model.to(device)
    model.eval()
    try :
        os.mkdir(os.path.join('./predictions'))
    except :
        pass
    # 이미지 전체 반복
    pred_xml = elemTree.Element('predictions')
    pred_xml.text = '\n  '
    for idx, data in enumerate(data_loader_test) :
        print('{} / {}'.format(idx+1, len(data_loader_test)))
        images, target = data
        images = list(image.to(device) for image in images)        
        outputs = model(images)
        output = outputs[0]
        masks, labels, scores = output['masks'], output['labels'], output['scores']
        texts = []
        # 이미지 한장에 대하여

        xml_image = elemTree.SubElement(pred_xml, 'image')
        xml_image.attrib['name'] = target[0]['image_id']
        xml_image.text = '\n    '

        for index in range(len(masks)) :
            mask, label, score = masks[index], int(labels[index]), scores[index]
            # class, score, x1, y1, x2, y2
            mask_arr = mask[0].cpu().detach().numpy()
            mask_bin = np.where(mask_arr > 0.3, True, False)
            polygons = Mask(mask_bin).polygons()
            points = polygons.points
            point = ''
            for p in points[0]:
                point += str(p[0]) + ',' + str(p[1]) +';'
            xml_predict = elemTree.SubElement(xml_image, 'predict')
            xml_predict.tail = '\n    '
            xml_predict.attrib['class_name'] = class_name[label]
            xml_predict.attrib['score'] = str(float(score))
            xml_predict.attrib['polygon'] = point
            if index == len(masks) - 1 :
                xml_predict.tail = '\n  '
        xml_image.tail = '\n  '
        if idx == len(data_loader_test) - 1:
            xml_image.tail = '\n'
    pred_xml = elemTree.ElementTree(pred_xml)
    pred_xml.write('./predictions/'+ predict_path + '.xml')
                
                
def train(model, data_loader, device, num_epochs, optimizer=None, lr_scheduler=None) :
    # 모델을 GPU나 CPU로 옮깁니다
    try :
        os.mkdir(os.path.join('./weights'))
    except :
        pass
    
    model.to(device)

    # 옵티마이저(Optimizer)를 만듭니다
    if optimizer == None :
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.001,
                                    momentum=0.9, weight_decay=0.0005)
        
    # 학습률 스케쥴러를 만듭니다
    if lr_scheduler == None :
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=40,gamma=0.1)

    for epoch in range(num_epochs):
        
        print(epoch)
        
        model.train()
        count = 0
        
        for images, targets in data_loader:
            count += len(images)
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())


            # reduce losses over all GPUs for logging purposes
            world_size = 1

            if dist.is_available() and dist.is_initialized() :
                world_size = dist.get_world_size()

            if world_size >= 2:
                with torch.no_grad():
                    names = []
                    values = []
                    # sort the keys so that they are consistent across processes
                    for k in sorted(loss_dict.keys()):
                        names.append(k)
                        values.append(loss_dict[k])
                    values = torch.stack(values, dim=0)
                    dist.all_reduce(values)
                    if average:
                        values /= world_size
                    loss_dict = {k: v for k, v in zip(names, values)}


            losses_reduced = sum(loss for loss in loss_dict.values())

            loss_value = losses_reduced.item()

            print('epoch {} [{}/{}]loss_classifier : {} loss_box_reg : {} loss_objectness : {} loss_rpn_box_reg : {}'.format(
                epoch, count, len(data_loader)*data_loader.batch_size, loss_dict['loss_classifier'], loss_dict['loss_box_reg'], loss_dict['loss_objectness'], loss_dict['loss_rpn_box_reg']
            ))

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(targets)
                print(loss_dict)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        save_model('./weights/{}'.format(epoch), model, optimizer, lr_scheduler)
        lr_scheduler.step()
        


def main():

    # mode argument
    args = argparse.ArgumentParser()
    args.add_argument("--num_classes", type=int, default=22)
    args.add_argument("--lr", type=int, default=0.005)
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--num_epochs", type=int, default=30)
    args.add_argument("--model_name", type=str, default="weights/0.pth")
    args.add_argument("--prediction_file", type=str, default="prediction")
    args.add_argument("--batch", type=int, default=16)
    args.add_argument("--mode", type=str, default="train")
    
    config = args.parse_args()

    num_classes = config.num_classes
    base_lr = config.lr
    cuda = config.cuda
    num_epochs = config.num_epochs
    model_name = config.model_name
    prediction_file = config.prediction_file
    batch = config.batch
    mode = config.mode

    # 도움 함수를 이용해 모델을 가져옵니다
    new_model = model.get_model_instance_segmentation(num_classes)

 # 학습을 GPU로 진행하되 GPU가 가용하지 않으면 CPU로 합니다
    device = torch.device('cuda') if cuda else torch.device('cpu')

    # 모델을 GPU나 CPU로 옮깁니다
    new_model.to(device)

    if mode == 'train':
        # 데이터셋과 정의된 변환들을 사용합니다
        '''dataset = CustomDataset(DATASET_PATH, dataloader.get_transform(train=True))
        dataset_val = CustomDataset(DATASET_PATH, dataloader.get_transform(train=False))
        # 데이터셋을 학습용과 테스트용으로 나눕니다(역자주: 여기서는 전체의 50개를 테스트에, 나머지를 학습에 사용합니다)
        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:-10])
        dataset_val = torch.utils.data.Subset(dataset_val, indices[-10:])
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=batch, shuffle=True,num_workers=4, collate_fn=dataloader.collate_fn)
    '''
        dataset =dataloader.make_dataset(DATASET_PATH)
        # 데이터 로더를 학습용과 검증용으로 정의합니다
        dataset_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch, shuffle=True, num_workers=0,collate_fn=dataloader.collate_fn)

        # 옵티마이저(Optimizer)를 만듭니다
        params = [p for p in new_model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=base_lr,
                                    momentum=0.9, weight_decay=0.0005)
            
        # 학습률 스케쥴러를 만듭니다
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
        
        train(new_model, dataset_loader, device, num_epochs, optimizer=optimizer, lr_scheduler=lr_scheduler)
    
    elif mode == 'test' :
        dataset_test =dataloader.make_testset('/tf/notebooks/08/data/08_road_conditon/test')

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, collate_fn=dataloader.collate_fn)
        
        load_model(model_name, new_model)
        test(new_model, data_loader_test, device, prediction_file)
    
        

    print("That's it!")

if __name__ == '__main__' :
    sys.setrecursionlimit(3000)
    main()