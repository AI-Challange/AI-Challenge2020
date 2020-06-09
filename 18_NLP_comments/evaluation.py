import argparse
import numpy as np
import sklearn.metrics

def evaluate(pred_bias_labels, pred_hate_labels, gt_bias_labels, gt_hate_labels):
    print('total :' + str(len(pred_bias_labels)) + ', none :' + str(pred_bias_labels.count('none')) + ', gender : ' + str(
        pred_bias_labels.count('gender')) + ', others : ' + str(pred_bias_labels.count('others')))
    print('total : ' + str(len(pred_hate_labels)) + ', none :' + str(pred_hate_labels.count('none')) + ', hate : ' + str(
        pred_hate_labels.count('hate')) + ', offensive : ' + str(pred_hate_labels.count('offensive')))


    f1_bias = sklearn.metrics.f1_score(gt_bias_labels,pred_bias_labels, average = 'weighted')
    f1_hate = sklearn.metrics.f1_score(gt_hate_labels, pred_hate_labels, average = 'weighted')
    print('f1_bias : ' + str(f1_bias))
    print('f1_hate : ' + str(f1_hate))

    mean_f1 = (f1_bias+f1_hate)/2
    return mean_f1


def read_prediction_file(file_name):
    with open(file_name, 'r') as f:
        bias_list = []
        hate_list = []
        for line in f.readlines()[0:]:
            v = line.strip().split('\t')
            bias_list.append(v[1])
            hate_list.append(v[2])
    return bias_list, hate_list


def read_test_file(file_name):
    with open(file_name, 'r') as f:
        bias_list = []
        hate_list = []
        for line in f.readlines()[0:]:
            v = line.strip().split('\t')
            bias_list.append(v[2])
            hate_list.append(v[3])
    return bias_list, hate_list


def evaluation_metrics(prediction_file, test_file):
    pred_bias_labels, pred_hate_labels = read_prediction_file(prediction_file)
    gt_bias_labels, gt_hate_labels = read_test_file(test_file)
    return evaluate(pred_bias_labels, pred_hate_labels, gt_bias_labels, gt_hate_labels)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--prediction_file', type=str, default='prediction.txt')
    args.add_argument('--test_file', type=str, default='test_label')

    config = args.parse_args()

    print(evaluation_metrics(config.prediction_file, config.test_file))