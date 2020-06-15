# Forecast Traffic

## Task
```
이전 기간동안 35개 도로의 교통량과, 예측 기간동안 10개 도로의 교통량을 기반으로 25개 도로의 교통량 예측

각 고속도로(전체 35개)의 1시간 단위 교통량 데이터 제공.
 - 10개 고속도로의 교통량 데이터는 전체 기간을 제공
 - 25개 고속도로의 교통량 데이터는 예측기간을 제외한 기간만 제공
```

## Dataset(Sample)
| Phase | # |
| - | - |
| train | 2020.01.01 ~ 05.01 |
| validate | 2020.04.17 ~ 05.16 |
| test | 2020.05.02 ~ 05.31 |

Blue Cell : 제공되는 데이터 / Red Cell : 제공되지 않는 데이터 (-999로 표기됨)

<img width=600 src="Image/Sample_Dataset.png"/>

```
****** 제공되는 Dataset은 Sample입니다. ******
- 제공되는 Sample Dataset은 Sequence Data임을 고려하여, 예측기간의 교통량을 예측 할 수 있도록 이전 기간의 데이터가 함께 제공됩니다.
- 이 문제에서는 참가자가 원하는 크기의 Window를 적용 할 수 있도록 Dataset들을 Concat 등으로 연결/조합하는 것이 허용됩니다.
- Evalution()은 예측기간의 Red Cell에 대해서만 평가를 진행합니다.
 (prediction.txt 파일의 최근 15일 예측값에 대해서 Target Value와 매칭하여 Evaluation)
*** 주의점 : Test Dataset을 Concat 등으로 확장하여 사용하는 경우, 사후 검증에서도 재현 될 수 있도록 제출 코드상에 자동화하여 작성 !!! 필수 !!!
            (Admin은 참가자에게 제공되는 Sample Dataset을 기준으로 Dataset 보유)

참고 : Train Dataset(2020. 01. 01 ~ 05. 01) 중 1일치 데이터(3월 30일)은 기계 오류로 인해 데이터가 수집 되지 않았음
```


## Data Directory
```
\_data
    \_ train.csv
    \_ validate.csv
    \_ test.csv
```

## Data Sample
<img width=800 src="Image/Sample_1.PNG"/>

참고 : Test Dataset의 예측 기간 중 25개 도로의 교통량 데이터는 -999로 처리되어있음
<img width=800 src="Image/Sample_2.PNG"/>


## Metric
```
평가를 위한 Metric : RMSLE
```
<img width=450 src="Image/RMSE_Custom.png"/>


## Commands
```
# train
python main.py 

# test 
python main.py --batch=4 --model_name="1.pth" --mode="test"


All options in example commands are default value.
```



