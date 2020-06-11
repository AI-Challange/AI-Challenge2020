import argparse
import numpy as np
import torch.nn as nn
import torch
from torch.utils import data
from PIL import Image
import os


def evaluate(prediction_dir, labels, cuda, validate_label):
    loss_fn = nn.L1Loss()
    if cuda:
        loss_fn = loss_fn.cuda()
    loss_array = None

    for index, [pred_image, answer_image] in enumerate(labels):

        pred_image = Image.open(os.path.join(prediction_dir, pred_image))
        answer_image = Image.open(os.path.join(os.path.dirname(validate_label), answer_image))

        pred_image = np.asarray(pred_image, dtype=float)
        answer_image = np.asarray(answer_image, dtype=float)

        pred_image = torch.from_numpy(pred_image)
        answer_image = torch.from_numpy(answer_image)

        if cuda:
            pred_image = pred_image.cuda()
            answer_image = answer_image.cuda()

        loss = loss_fn(pred_image, answer_image)
        if cuda:
            loss = np.expand_dims(loss.detach().cpu(), axis=0)

        if index == 0:
            loss_array = loss
        else:
            loss_array = np.concatenate((loss_array, loss), axis=0)

    avg = np.mean(loss_array, axis=0)
    return avg


def read_validate_label(file_name):
    with open(file_name, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    pairs = [l.strip().split(',') for l in lines]
    return pairs


def evaluation_metrics(prediction_dir, validate_label, cuda):
    v_labels = read_validate_label(validate_label)

    return evaluate(prediction_dir, v_labels, cuda, validate_label)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--prediction_dir', type=str, default='prediction')
    args.add_argument('--validate_label', type=str, default='data/validate/validate_labels.csv')
    args.add_argument('--cuda', type=bool, default=True)

    config = args.parse_args()

    print(evaluation_metrics(config.prediction_dir, config.validate_label, config.cuda))