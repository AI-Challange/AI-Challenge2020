import dataloader
from dataloader import CustomDataset
import model
import evaluation
import torch
import torch.distributed as dist
import math
import argparse
import os
import sys
import xml.etree.ElementTree as elemTree

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

model_dir = 'saved_model'

def make_folder(path) :
    try :
        os.mkdir(os.path.join(path))
    except :
        pass

def save_model(model_name, model, optimizer, scheduler):
    make_folder(model_dir)
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }

    torch.save(state, os.path.join(model_dir, model_name + '.pth'))
    print('model saved')
    
def load_model(model_name, model, optimizer=None, scheduler=None):
    state = torch.load(os.path.join(model_dir, model_name + '.pth'))
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    print('model loaded')
    
def test(model, data_loader_test, device, prediction_dir) :
    class_name = {1 : 'bus', 2 : 'car', 3 :'carrier', 4 : 'cat', 5 : 'dog', 
                  6 : 'motorcycle', 7 : 'movable_signage', 8 : 'person', 9 : 'scooter', 10 : 'stroller', 
                  11 : 'truck', 12 : 'wheelchair', 13 : 'barricade', 14 : 'bench', 15 : 'chair',
                  16 : 'fire_hydrant', 17 : 'kiosk', 18 : 'parking_meter', 19 : 'pole', 20 : 'potted_plant', 
                  21 : 'power_controller', 22 : 'stop', 23 : 'table', 24 : 'traffic_light_controller', 
                  25 : 'traffic_sign', 26 : 'tree_trunk', 27 : 'bollard', 28 : 'bicycle'}
    model.to(device)
    model.eval()

    make_folder(prediction_dir)
    
    # 이미지 전체 반복
    pred_xml = elemTree.Element('predictions')
    pred_xml.text = '\n  '
    batch_size = data_loader_test.batch_size
    for idx, data in enumerate(data_loader_test) :
        images, target = data
        images = list(image.to(device) for image in images)
        count = len(images)       
        outputs = model(images)
        output = outputs[0]
        boxes, labels, scores = output['boxes'], output['labels'], output['scores']
        texts = []
        # 이미지 한장에 대하여
        for n in range(count) :
                
            xml_image = elemTree.SubElement(pred_xml, 'image')
            
            img_name = data_loader_test.dataset.imgs[idx*batch_size+n].split('/')[-1].split('.png')[0]
            img_name = img_name.split('.jpg')[0]
            xml_image.attrib['name'] = img_name
            xml_image.text = '\n    '

            for index in range(len(boxes)) :
                box, label, score = boxes[index], int(labels[index]), scores[index]
                # class, score, x1, y1, x2, y2
                xml_predict = elemTree.SubElement(xml_image, 'predict')
                xml_predict.tail = '\n    '
                xml_predict.attrib['class_name'] = class_name[label]
                xml_predict.attrib['score'] = str(float(score))
                xml_predict.attrib['x1'] = str(int(box[0]))
                xml_predict.attrib['y1'] = str(int(box[1]))
                xml_predict.attrib['x2'] = str(int(box[2]))
                xml_predict.attrib['y2'] = str(int(box[3]))
                if index == len(boxes) - 1 :
                    xml_predict.tail = '\n  '
            xml_image.tail = '\n  '
            if idx == len(data_loader_test) - 1 and n == (count - 1):
                xml_image.tail = '\n'
    pred_xml = elemTree.ElementTree(pred_xml)
    pred_xml.write(prediction_dir + '/predictions.xml')

def train(model, data_loader_train, data_loader_val, device, num_epochs, prediction_dir, print_iter, optimizer=None, lr_scheduler=None) :
    # 모델을 GPU나 CPU로 옮깁니다
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
        
        for images, targets in data_loader_train:
            count += len(images)
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            world_size = 1

            if dist.is_available() and dist.is_initialized() :
                world_size = dist.get_world_size()

            if world_size >= 2:
                with torch.no_grad():
                    names = []
                    values = []
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
            if count % print_iter < data_loader_train.batch_size :
                print('epoch {} [{}/{}] loss_classifier : {} loss_box_reg : {} loss_objectness : {} loss_rpn_box_reg : {}'.format(
                    epoch, count, len(data_loader_train.dataset), loss_dict['loss_classifier'], loss_dict['loss_box_reg'], loss_dict['loss_objectness'], loss_dict['loss_rpn_box_reg']
                ))

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(targets)
                print(loss_dict)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        validation(model, data_loader_val, device, prediction_dir)
                
        save_model('{}'.format(epoch), model, optimizer, lr_scheduler)

        lr_scheduler.step()

def validation(model, data_loader_val, device, prediction_dir) :
    test(model, data_loader_val, device, prediction_dir)
    GT_PATH = os.path.join(DATASET_PATH, 'val')
    DR_PATH = os.path.join(prediction_dir,'predictions.xml')
    res = evaluation.evaluation_metrics(GT_PATH, DR_PATH)
    print('validation : ', res)
    return res

try:
    from nipa import nipa_data
    DATASET_PATH = nipa_data.get_data_root('deepfake')
except:
    DATASET_PATH = os.path.join('./data')

def main():

    # mode argument
    args = argparse.ArgumentParser()
    args.add_argument("--num_classes", type=int, default=29)
    args.add_argument("--lr", type=float, default=0.001)
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--num_epochs", type=int, default=30)
    args.add_argument("--model_name", type=str, default="19")
    args.add_argument("--batch", type=int, default=2)
    args.add_argument("--mode", type=str, default="train")
    args.add_argument("--prediction_dir", type=str, default="prediction")
    args.add_argument("--print_iter", type=int, default=10)

    config = args.parse_args()

    num_classes = config.num_classes
    base_lr = config.lr
    cuda = config.cuda
    num_epochs = config.num_epochs
    model_name = config.model_name
    batch = config.batch
    mode = config.mode
    prediction_dir = config.prediction_dir
    print_iter = config.print_iter

    # 도움 함수를 이용해 모델을 가져옵니다
    new_model = model.get_model_instance_segmentation(num_classes)
    
    #check parameter of model
    print("------------------------------------------------------------")
    total_params = sum(p.numel() for p in new_model.parameters())
    print("num of parameter : ",total_params)
    trainable_params = sum(p.numel() for p in new_model.parameters() if p.requires_grad)
    print("num of trainable_ parameter :",trainable_params)
    print("------------------------------------------------------------")

    # 학습을 GPU로 진행하되 GPU가 가용하지 않으면 CPU로 합니다
    device = torch.device('cuda') if cuda else torch.device('cpu')

    if mode == 'train':
        # 데이터셋과 정의된 변환들을 사용합니다
        data_loader_train = dataloader.data_loader(DATASET_PATH, batch, phase='train')
        data_loader_val = dataloader.data_loader(DATASET_PATH, 1, phase='val')

        # 옵티마이저(Optimizer)를 만듭니다
        params = [p for p in new_model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=base_lr,
                                    momentum=0.9, weight_decay=0.0005)
            
        # 학습률 스케쥴러를 만듭니다
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=40,gamma=0.1)
        
        train(new_model, data_loader_train, data_loader_val, device, num_epochs, prediction_dir, print_iter, optimizer=optimizer, lr_scheduler=lr_scheduler)
        
    
    elif mode == 'test' :
        load_model(model_name, new_model)
        data_loader_test = dataloader.data_loader(DATASET_PATH, 1, phase='test')        
        test(new_model, data_loader_test, device, prediction_dir)

    print("That's it!")

if __name__ == '__main__' :
    main()
