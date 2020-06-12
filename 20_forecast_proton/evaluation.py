# %%
import math
import pandas as pd
import numpy as np
import argparse
#0 미만은 계산하지 않는다.


def RMSE(gt_value, pred_value, length):
    weight = [1,68,182, 809,6041]
    sum_error = 0
    pass_num = 0
    for i in range(length):
        if gt_value[i] < 0 :
            pass_num += 1
            continue
        if gt_value[i] < 10:
            sum_error += (gt_value[i] - pred_value[i]) ** 2 * weight[0]
        elif (gt_value[i] > 10) & (gt_value[i] < 100):
            sum_error += (gt_value[i] - pred_value[i]) ** 2 * weight[1]
        elif (gt_value[i] > 100) & (gt_value[i] < 1000):
            sum_error += (gt_value[i] - pred_value[i]) ** 2 * weight[2]
        elif (gt_value[i] > 1000) & (gt_value[i] < 10000):
            sum_error += (gt_value[i] - pred_value[i]) ** 2 * weight[3]
        elif (gt_value[i] > 10000):
            sum_error += (gt_value[i] - pred_value[i]) ** 2 * weight[4]
    sum_error = float(sum_error / length)
    sum_error = sum_error ** 0.5 + 1
    sum_error = math.log(sum_error)
    return sum_error

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--gt", type=str, default='./data/test_proton_admin.csv')
    args.add_argument("--pred", type=str, default='./prediction/predict.csv')
    
    config = args.parse_args()
    
    gt = pd.read_csv(config.gt)['proton']
    pr = pd.read_csv(config.pred)['predict']
    print(len(gt), len(pr))
    result = RMSE(gt, pr, len(gt))
    
    print('evaluation: ', result)
    
if __name__ == '__main__' :
    main()
