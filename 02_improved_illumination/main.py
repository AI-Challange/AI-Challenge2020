import os
import math
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
import argparse
import shutil

from dataloader import data_loader
from evaluation import evaluation_metrics
from model import Net


try:
    from nipa import nipa_data
    DATASET_PATH = nipa_data.get_data_root('improving_illumination')
except:
    DATASET_PATH = os.path.join('./data')

def feed_infer(prediction_dir, model, data_loader, cuda):
    res_out = None
    res_name = None
    
    if os.path.isdir(prediction_dir):
        shutil.rmtree(prediction_dir)

    try:
        os.mkdir(prediction_dir)
    except Exception as e:
        print("Failed to make directory '{}'".format(prediction_dir))
        raise e
    
    print('infer & write output')

    for index, (input_names, inputs, _, _) in enumerate(data_loader):
        if cuda:
            inputs = inputs.cuda()
            
        out = model(inputs)
        out = out.detach().cpu()

        res_name = list(input_names)
        res_out = out

        for image, name in zip(res_out, res_name):
            save_image(image, os.path.join(prediction_dir, name))


def validate(prediction_dir, model, validate_dataloader, validate_label_file, cuda):
    feed_infer(prediction_dir, model, validate_dataloader, cuda)
    
    metric_result = evaluation_metrics(prediction_dir, validate_label_file, cuda)
    print('Eval result: {:.4f}'.format(metric_result))
    return metric_result


def test(prediction_dir, model, test_dataloader, cuda):
    feed_infer(prediction_dir, model, test_dataloader, cuda)


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
    args.add_argument("--lr", type=float, default=0.001)
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--num_epochs", type=int, default=100)
    args.add_argument("--print_iter", type=int, default=10)
    args.add_argument("--model_name", type=str, default="model.pth")
    args.add_argument("--prediction_dir", type=str, default="prediction")
    args.add_argument("--batch", type=int, default=4)
    args.add_argument("--mode", type=str, default="train")

    config = args.parse_args()

    base_lr = config.lr
    cuda = config.cuda
    num_epochs = config.num_epochs
    print_iter = config.print_iter
    model_name = config.model_name
    prediction_dir = config.prediction_dir
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

        
        #check parameter of model
        print("------------------------------------------------------------")
        total_params = sum(p.numel() for p in model.parameters())
        print("num of parameter :", total_params)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("num of trainable_ parameter :", trainable_params)
        print("------------------------------------------------------------")
        
        # train
        for epoch in range(num_epochs):
            model.train()
            for iter_, train_data in enumerate(train_dataloader):
                # fetch train data
                _, inputs, _, answers = train_data
                if cuda:
                    inputs = inputs.cuda()
                    answers = answers.cuda()

                # update weight
                pred = model(inputs)
                loss = loss_fn(pred, answers)
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
            validate(prediction_dir, model, validate_dataloader, validate_label_file, cuda)

            time_ = datetime.datetime.now()
            elapsed = datetime.datetime.now() - time_
            print('[epoch {}] elapsed: {}'.format(epoch + 1, elapsed))

    elif mode == 'test':
        model.eval()
        # get data loader
        test_dataloader, _ = data_loader(root=DATASET_PATH, phase='test', batch_size=batch)
        test(prediction_dir, model, test_dataloader, cuda)
        # submit test result
        
