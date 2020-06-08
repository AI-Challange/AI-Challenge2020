import argparse
import numpy as np


def evaluate(prediction_labels, gt_labels):
    num_data = len(gt_labels)
    hamming_sum = 0.0
    
    for index, query in enumerate(gt_labels):
        gt_label = [int(x) for x in gt_labels[query]]
        pred_label = [int(x) for x in prediction_labels[query]]
        
        correct = [gt_label[0] == pred_label[0], gt_label[1] == pred_label[1]].count(True)
        plant_xor = 1.0 if (gt_label[0] != pred_label[0]) else 0
        disease_xor = 1.0 if (gt_label[1] != pred_label[1]) else 0
        
        hamming_sum += (plant_xor + disease_xor)

    hamming_loss = hamming_sum / num_data / 2
    print('hamming_loss:', hamming_loss)
    return hamming_loss


def read_prediction_file(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        
    dictionary = {}
    
    for line in lines:
        [name, plant, disease] = line.replace('\n', '').split(' ')
        dictionary[name] = (plant, disease)
        
    return dictionary


def read_test_file(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        
    dictionary = {}
    
    for line in lines:
        [name, plant, disease] = line.replace('\n', '').split(' ')
        dictionary[name] = (plant, disease)
        
    return dictionary


def evaluation_metrics(prediction_file, test_file):
    prediction_labels = read_prediction_file(prediction_file)
    gt_labels = read_test_file(test_file)

    return evaluate(prediction_labels, gt_labels)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--prediction_file', type=str, default='prediction.txt')
    args.add_argument('--test_file', type=str, default='./data/validate/validate_labels.txt')

    config = args.parse_args()

    print(evaluation_metrics(config.prediction_file, config.test_file))