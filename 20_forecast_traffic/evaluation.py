import math
import pandas as pd
import numpy as np
import argparse


def RMSLE(gt_value, pred_value):
    sum_error = 0
    length = len(gt_value)
    for i in range(length):
        sum_error += (math.log(gt_value[i] + 1) - math.log(pred_value[i] + 1)) ** 2
    sum_error = float(sum_error / length)
    sum_error = sum_error ** 0.5
    return sum_error


def read_test_file(file_name):
    label_index_pool = [2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 26, 27, 28, 30, 32, 33, 34]
    f =  open(file_name, 'r', encoding='utf-8-sig')
    lines = f.readlines()
    
    result = []
    for line in lines[-360:]:
        line = line.strip().split(',')
        for idx in label_index_pool:
            result.append(float(line[idx + 2]))

    f.close()
    return result


def read_prediction_file(file_name):
    f =  open(file_name, 'r', encoding='utf-8-sig')
    lines = f.readlines()
    
    result = []
    for line in lines:
        line = [float(x) for x in line.strip().split()]
        result.extend(line)
    
    f.close()
    return result


def evaluation_metrics(prediction_file, test_file):
    prediction_labels = read_prediction_file(prediction_file)
    gt_labels = read_test_file(test_file)

    return RMSLE(prediction_labels, gt_labels)    


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--prediction_file', type=str, default='prediction.txt')
    args.add_argument('--test_file', type=str, default='data/validate.csv')

    config = args.parse_args()

    print(evaluation_metrics(config.prediction_file, config.test_file))