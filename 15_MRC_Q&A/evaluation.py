import argparse
import json
import numpy as np


def edit_distance(r, h):
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8).reshape((len(r)+1, len(h)+1))
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
    d = edit_distance(r, h)
    # print the result in aligned way
    result = float(d[len(r)][len(h)]) / len(r) * 100
    return result


def evaluate(predictions, tests):
    wer_result = 0
    for qid, ground_truth in tests.items():
        prediction = predictions[qid]
        wer_result += wer(ground_truth.split(), prediction.split())

    wer_result = wer_result / len(predictions)
    return wer_result


def read_prediction_file(file_name):
    with open(file_name, encoding='utf-8-sig') as prediction_file:
        predictions = json.load(prediction_file)
    return predictions


def read_test_file(file_name):
    with open(file_name, encoding='utf-8-sig') as f:
        tests = json.load(f)

    tests = dict([(item['id'], item['answers'][0]['text']) for topic in tests['data'] for item in topic['paragraphs'][0]['qas']])

    return tests


def evaluation_metrics(prediction_file, test_file):
    prediction_labels = read_prediction_file(prediction_file)
    gt_labels = read_test_file(test_file)

    return evaluate(prediction_labels, gt_labels)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--prediction_file', type=str, default='prediction.json')
    args.add_argument('--test_file', type=str, default='data/validate.json')

    config = args.parse_args()

    print(evaluation_metrics(config.prediction_file, config.test_file))