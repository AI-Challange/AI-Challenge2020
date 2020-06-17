import os
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import argparse
import json

from dataloader import data_loader
from evaluation import evaluation_metrics
from model import Net

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

    DATASET_PATH = nipa_data.get_data_root('mrc')
except:
    DATASET_PATH = os.path.join('./data')


def _infer(model, cuda, data_loader):
    res_pred = []
    res_qid = []
    for _, d in enumerate(data_loader):
        seqs = d['sequence']
        qids = d['id']
        contexts = d['context']

        if cuda:
            seqs = seqs.type(torch.FloatTensor).cuda()

        for qid, sequence, context in zip(qids, seqs, contexts):
            pred = model(sequence)

            # variable pred includes start and end position of answer
            # answer_start = int(pred[0])
            # answer_end = int(pred[1])

            answer_start = 2
            answer_end = 6

            pred_answer = context[answer_start:answer_end]

            res_pred.append(pred_answer)
            res_qid.append(qid)

    return [res_qid, res_pred]


def feed_infer(output_file, infer_func):
    qids, preds = infer_func()

    print('write output')
    dic = {}
    for qid, pred in zip(qids, preds):
        dic[qid] = pred
    with open(output_file, 'w', encoding='utf-8-sig') as f:
        json.dump(dic, f, ensure_ascii=False)

    if os.stat(output_file).st_size == 0:
        raise AssertionError('output result of inference is nothing')


def validate(prediction_file, model, validate_dataloader, validate_label_file, cuda):
    feed_infer(prediction_file, lambda: _infer(model, cuda, data_loader=validate_dataloader))

    metric_result = evaluation_metrics(prediction_file, validate_label_file)
    print('Eval result: {:.4f}'.format(metric_result))
    return metric_result


def test(prediction_file_name, model, test_dataloader, cuda):
    feed_infer(prediction_file, lambda: _infer(model, cuda, data_loader=test_dataloader))


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


if __name__ == '__main__':
    # mode argument
    args = argparse.ArgumentParser()
    args.add_argument("--lr", type=int, default=0.001)
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--num_epochs", type=int, default=100)
    args.add_argument("--print_iter", type=int, default=10)
    args.add_argument("--model_name", type=str, default="model.pth")
    args.add_argument("--prediction_file", type=str, default="prediction.json")
    args.add_argument("--batch", type=int, default=4)
    args.add_argument("--mode", type=str, default="train")

    config = args.parse_args()

    base_lr = config.lr
    cuda = config.cuda
    num_epochs = config.num_epochs
    print_iter = config.print_iter
    model_name = config.model_name
    prediction_file = config.prediction_file
    batch = config.batch
    mode = config.mode

    # create model
    model = Net()

    if mode == 'test':
        load_model(model_name, model)

    if cuda:
        model = model.cuda()

    if mode == 'train':
        # define loss function
        loss_fn = nn.L1Loss()
        if cuda:
            loss_fn = loss_fn.cuda()

        # set optimizer
        optimizer = Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=base_lr, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=40, gamma=0.1)

        # get data loader
        train_dataloader, _ = data_loader(root=DATASET_PATH, phase='train', batch_size=batch)
        validate_dataloader, validate_label_file = data_loader(root=DATASET_PATH, phase='validate', batch_size=batch)
        time_ = datetime.datetime.now()
        num_batches = len(train_dataloader)

        # check parameter of model
        print("------------------------------------------------------------")
        total_params = sum(p.numel() for p in model.parameters())
        print("num of parameter : ", total_params)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("num of trainable_ parameter :", trainable_params)
        print("------------------------------------------------------------")

        # train
        for epoch in range(num_epochs):
            model.train()
            for iter_, train_data in enumerate(train_dataloader):

                input_seq = train_data['sequence'].type(torch.FloatTensor)
                answer_pos = train_data['answer_pos'].type(torch.FloatTensor)

                if cuda:
                    answer_pos = answer_pos.type(torch.FloatTensor).cuda()
                    input_seq = input_seq.type(torch.FloatTensor).cuda()

                # update weight
                pred = model(input_seq)
                loss = loss_fn(pred, answer_pos)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (iter_ + 1) % print_iter == 0:
                    elapsed = datetime.datetime.now() - time_
                    expected = elapsed * (num_batches / print_iter)
                    _epoch = epoch + ((iter_ + 1) / num_batches)
                    print('[{:.3f}/{:d}] loss({}) '
                          'elapsed {} expected per epoch {}'.format(
                        _epoch, num_epochs, loss.item(), elapsed, expected))
                    time_ = datetime.datetime.now()

            # scheduler update
            scheduler.step()

            # save model
            save_model(str(epoch + 1), model, optimizer, scheduler)

            # validate
            validate(prediction_file, model, validate_dataloader, validate_label_file, cuda)

            elapsed = datetime.datetime.now() - time_
            print('[epoch {}] elapsed: {}'.format(epoch + 1, elapsed))

    elif mode == 'test':
        model.eval()
        # get data loader
        test_dataloader, _ = data_loader(root=DATASET_PATH, phase='test', batch_size=batch)
        test(prediction_file, model, test_dataloader, cuda)
        # submit test result
