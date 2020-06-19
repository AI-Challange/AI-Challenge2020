import argparse
import numpy as np
from openpyxl import load_workbook

#Metric 작성
def evaluate(prediction_labels, gt_labels, max_vector):
    count = 0
    total_len = 0
    for index, query in enumerate(gt_labels):
        if '\ufeff' in gt_labels[query] : 
            gt_labels[query] = gt_labels[query].replace('\ufeff', '')
        gt_len = len(gt_labels[query])
        if '\ufeff' in prediction_labels[query] : 
            prediction_labels[query] = prediction_labels[query].replace('\ufeff', '')
        pred_len = len(prediction_labels[query])

        max_len = gt_len if gt_len >= pred_len else pred_len
        total_len += max_len

        min_len = pred_len if gt_len >= pred_len else gt_len

        for i in range(0, min_len) :
            if gt_labels[query][i] == prediction_labels[query][i] :
                count +=1

    f1 = count / total_len
    return f1


def read_prediction_pt(file_name):
    name = []
    pred = []
    dictionary = dict()
    with open(file_name, 'r', encoding='utf8-8-sig') as f:
        for line in f.readlines()[0:]:
            v = line.strip().split(' ', maxsplit = 1)
            name.append(v[0])   
            pred.append(v[1])
            dictionary[v[0]] = v[1]
    #([l.replace('\n', '').split(' ') for l in lines])
    #print(dictionary)
    return dictionary


def read_prediction_gt(file_name):
    with open(file_name, 'r', encoding= 'utf-8-sig') as f:
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
    #testset_path = '/data/7_icls_face/test/test_label'

    print(evaluation_metrics(config.prediction, testset_path, max_vector))
