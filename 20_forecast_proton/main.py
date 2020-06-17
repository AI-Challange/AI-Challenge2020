import argparse
import numpy as np
import glob
import evaluation
import dataloader
import torch
from torch import nn
import math
import sys
import os
from model import LinearRegression
from torch.nn import L1Loss as MAELoss
import pandas as pd

'''
컨테이너 내 폴더 설명
- /datasets : read only 폴더
- /tf/notebooks :  read/write 폴더
1. 참가자는 각 문제별로 데이터를 로드하기 위해 적절한 path를 하단에 입력해야합니다. (datasets/각 문제 폴더)
2. 참가자는 모델의 결과 파일(Ex> prediction.txt)을 write가 가능한 폴더에 저장되도록 적절 한 path를 입력해야합니다. (tf/notebooks)
3. 명시된 폴더 외에는 세션/컨테이너 등 재시작시 삭제될 수 있으니 참가자는 적절한 폴더에 Source code와 결과 파일 등을 저장해야합니다.
'''

try:
    from nipa import nipa_data
    DATASET_PATH = nipa_data.get_data_root('deepfake')
except:
    DATASET_PATH = os.path.join('./data')

# 음수의 정답을 가진 데이터를 제외합니다.
def handler(datas, labels, device) :
    new_datas = []
    new_labels = []
    for i in range(len(labels)) :
        if labels[i] >= 0 :
            new_datas.append(datas[i].to(device))
            new_labels.append(labels[i].to(device))
    return (new_datas, new_labels)

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

def train(num_epochs, model, device, train_loader, val_loader, optimizer, lr_scheduler, prediction_dir, print_iter) :
    criterion = MAELoss()
    criterion.to(device)
    model.to(device)
    for epoch in range(num_epochs) :
        print(epoch)
        count = 0
        for i, datas in enumerate(train_loader) :
            datas, labels = datas
            datas, labels = handler(datas, labels, device)
            for j in range(len(datas)) :
                pred = model(datas[j])
                loss = criterion(pred, labels[j])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if count % print_iter == 0 :
                    print('epoch : {} [{}/{}], loss : {}'.format(epoch, count, len(train_loader), loss))
                count += 1
        validation(model, device, val_loader, prediction_dir)
        save_model('{}'.format(epoch), model, optimizer, lr_scheduler)
        
        lr_scheduler.step()

def test(model, device, test_loader, prediction_dir) :
    model.to(device)
    model.eval()
    make_folder(prediction_dir)
    time_list = test_loader.dataset.labels[:,0].reshape(-1,1)
    pred_list = []
    for i, datas in enumerate(test_loader) :
        datas, labels = datas
        datas = list(data.to(device) for data in datas)
        pred = model(datas[0])
        pred_list.append(float(pred[0]))
    pred_list = np.array(pred_list).reshape(-1,1)
    res = np.append(time_list, pred_list, axis=1)
    res_df = pd.DataFrame(res, columns=['time tag', 'predict'])
    res_df.to_csv(os.path.join(prediction_dir, 'predict.csv'), index=None)
    return res_df

def validation(model, device, val_loader, prediction_dir) :
    ground_truths = val_loader.dataset.labels[:,1]
    preds = test(model, device, val_loader, prediction_dir)['predict']
    print('validation test finish')
    res = evaluation.RMSE(ground_truths, preds, len(ground_truths))
    print('validation : ', res)

def main() :
    # mode argument
    args = argparse.ArgumentParser()
    args.add_argument("--lr", type=float, default=0.0001)
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--num_epochs", type=int, default=50000)
    args.add_argument("--model_name", type=str, default="5")
    args.add_argument("--batch", type=int, default=2)
    args.add_argument("--mode", type=str, default="train")
    args.add_argument("--prediction_dir", type=str, default="prediction")
    args.add_argument("--print_iter", type=int, default=20000)
    
    config = args.parse_args()

    lr = config.lr
    cuda = config.cuda
    num_epochs = config.num_epochs
    model_name = config.model_name
    batch = config.batch
    mode = config.mode
    prediction_dir = config.prediction_dir
    print_iter = config.print_iter
    nIn = 12
    nOut = 1

    model = LinearRegression(nIn, nOut)
    device = torch.device('cuda') if cuda else torch.device('cpu')

    #check parameter of model
    print("------------------------------------------------------------")
    total_params = sum(p.numel() for p in model.parameters())
    print("num of parameter : ",total_params)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("num of trainable_ parameter :",trainable_params)
    print("------------------------------------------------------------")

    if mode == 'train' :
        print('train start')
        train_loader = dataloader.data_loader(DATASET_PATH, batch, phase = 'train')
        val_loader = dataloader.data_loader(DATASET_PATH, 1, phase = 'val')
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params=params, lr=lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=40,gamma=0.1)
        train(num_epochs, model, device, train_loader, val_loader, optimizer, lr_scheduler, prediction_dir, print_iter)
    elif mode == 'test' :
        print('test start')
        test_loader = dataloader.data_loader(DATASET_PATH, 1, phase = 'test')
        load_model(model_name, model)
        test(model, device, test_loader, prediction_dir)

if __name__ == '__main__' :
    main()
