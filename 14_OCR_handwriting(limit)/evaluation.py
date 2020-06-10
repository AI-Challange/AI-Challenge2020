import glob
import json
import numpy
import argparse


def editDistance(r, h):

    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        d[i][0] = i
    for j in range(len(h)+1):
        d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d

def wer(r, h):

    # build the matrix
    d = editDistance(r, h)

    # print the result in aligned way
    result = float(d[len(r)][len(h)]) / len(r) * 100
    return result
    
def evaluate(prediction_path, label_path) :
    
    predicts = read_prediction_file(prediction_path)
    ground_truths = read_test_file(label_path)

    total_wer = 0
   
    for i in range(len(predicts)) :
        wer_val = wer(ground_truths[i]['text'].split(), predicts[i]['prediction'].split())
        total_wer += wer_val
    ret = total_wer / len(predicts)
    print('val : {}'.format(ret))
    return ret

def read_prediction_file(prediction_path):
    with open(prediction_path) as f:
        predict_file = json.load(f)
    predicts = sorted(predict_file['predict'], key=lambda x: x['image_path'])
    return predicts
    

def read_test_file(label_path):
    with open(label_path) as f:
        label_file = json.load(f)
    ground_truths = sorted(label_file['annotations'], key=lambda x: x['file_name'])
    return ground_truths
    

def evaluation_metrics(prediction_path, label_path):
    return evaluate(prediction_path, label_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--prediction_path', type=str, default='./prediction_result/predict6.json')
    args.add_argument('--label_path', type=str, default='./data/val/val.json')

    config = args.parse_args()

    print(evaluation_metrics(config.prediction_path, config.label_path))