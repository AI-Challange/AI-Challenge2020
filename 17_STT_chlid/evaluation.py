import argparse
import numpy as np
from openpyxl import load_workbook


def evaluate(prediction_labels, gt_labels, max_vector):
    count = 0.0
    for index, query in enumerate(gt_labels):
        if gt_labels[query] == prediction_labels[query]:
            count += 1.0

    acc = count / float(len(prediction_labels))
    return acc


def read_prediction_pt(file_name):
    name = []
    pred = []
    dictionary = dict()
    with open(file_name, 'r', encoding='utf8') as f:
        for line in f.readlines()[0:]:
            v = line.strip().split(' ', maxsplit = 1)
            name.append(v[0])   
            pred.append(v[1])
            dictionary[v[0]] = v[1]
    return dictionary


def read_prediction_gt(file_name):
    with open(file_name, 'r', encoding='utf8') as f:
        lines = f.readlines()
    dictionary = dict([l.replace('\n', '').split(' ') for l in lines])
    return dictionary


def evaluation_metrics(prediction_file, testset_path, max_vector):
    prediction_labels = read_prediction_pt(prediction_file)
    gt_labels = read_prediction_gt(testset_path)

    return evaluate(prediction_labels, gt_labels, max_vector)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--prediction', type=str, default='pred.txt')
    config = args.parse_args()

    print(evaluation_metrics(config.prediction, testset_path))
