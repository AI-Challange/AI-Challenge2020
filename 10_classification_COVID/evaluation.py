import argparse
import numpy as np
from openpyxl import load_workbook

#Metric 작성
def evaluate(prediction_labels, gt_labels):
    count = 0.0
    for index, query in enumerate(gt_labels):
        gt_label = int(gt_labels[query])
        pred_label = int(prediction_labels[query])

        if gt_label == pred_label:
            count += 1.0

    acc = count / float(len(gt_labels))
    return acc


def read_prediction_pt(file_name):
    with open(file_name) as f:
        lines = f.readlines()
    dictionary = dict([l.replace('\n', '').split(' ') for l in lines])
    return dictionary


def read_prediction_gt(file_name):

    
    with open(file_name) as f:
        lines = f.readlines()
    dictionary = dict([l.replace('\n', '').split(' ') for l in lines])
    return dictionary


def evaluation_metrics(prediction_file, testset_path):
    prediction_labels = read_prediction_pt(prediction_file)
    gt_labels = read_prediction_gt(testset_path)

    return evaluate(prediction_labels, gt_labels)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--prediction', type=str, default='pred.txt')
    config = args.parse_args()
    #testset_path = '/data/7_icls_face/test/test_label'

    print(evaluation_metrics(config.prediction, testset_path))
