import argparse
import re
from collections import Counter
import json
from bs4 import BeautifulSoup
import string

'''본 스크립트는 KorQuAD 2.0 평가 스크립트를 바탕으로 작성됨.'''

def normalize_answer(s):    
    def tag_clean(t):
        return BeautifulSoup(t, "lxml").get_text()

    def remove_(text):
        ''' 불필요한 기호 제거 '''
        text = re.sub("'", " ", text)
        text = re.sub('"', " ", text)
        text = re.sub('《', " ", text)
        text = re.sub('》', " ", text)
        text = re.sub('<', " ", text)
        text = re.sub('>', " ", text) 
        text = re.sub('〈', " ", text)
        text = re.sub('〉', " ", text)   
        text = re.sub("\(", " ", text)
        text = re.sub("\)", " ", text)
        text = re.sub("‘", " ", text)
        text = re.sub("’", " ", text)      
        return text

    def white_space_fix(text):
        return ' '.join(text.split()).replace('\n','').replace('\t','').replace(' ','')

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(remove_(tag_clean(s)))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    #F1 by character
    prediction_Char = []
    for tok in prediction_tokens:
        now = [a for a in tok]
        prediction_Char.extend(now)

    ground_truth_Char = []
    for tok in ground_truth_tokens:
        now = [a for a in tok]
        ground_truth_Char.extend(now)   

    common = Counter(prediction_Char) & Counter(ground_truth_Char)
    num_same = sum(common.values())
    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_Char)
    recall = 1.0 * num_same / len(ground_truth_Char)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def evaluate(predictions, tests):
    f1 = 0
    for qid, ground_truth in tests.items():
        prediction = predictions[qid]
        f1 += f1_score(prediction, ground_truth)

    f1 = 100.0 * f1 / len(predictions)
    return f1


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