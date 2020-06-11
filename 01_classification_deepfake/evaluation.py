import argparse
import numpy as np
import sklearn.metrics


def evaluate(prediction_labels, gt_labels):
    count = 0.0
    gt_list = []
    prediction_list = []
    for index, query in enumerate(gt_labels):
        gt_label = int(gt_labels[query])
        pred_label = int(prediction_labels[query])
        gt_list.append(gt_label)
        prediction_list.append(pred_label)

    f1 = sklearn.metrics.f1_score(gt_list, prediction_list)
    return f1


def read_prediction_file(file_name):
    with open(file_name) as f:
        lines = f.readlines()
    dictionary = dict([l.replace('\n', '').split(' ') for l in lines])
    return dictionary


def read_test_file(file_name):
    with open(file_name) as f:
        lines = f.readlines()
    dictionary = dict([l.replace('\n', '').split(' ') for l in lines])
    return dictionary


def evaluation_metrics(prediction_file, test_file):
    prediction_labels = read_prediction_file(prediction_file)
    gt_labels = read_test_file(test_file)

    return evaluate(prediction_labels, gt_labels)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--prediction_file', type=str, default='prediction.txt')
    args.add_argument('--test_file', type=str, default='test_label')

    config = args.parse_args()

    print(evaluation_metrics(config.prediction_file, config.test_file))