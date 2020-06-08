# Improved Illumination

## Task
Input : 흉부 CT 이미지 / 
Output : COVID 양성/음성

## Dataset
| Phase | # |
| - | - |
| train | 546 |
| validate | 100 |
| test | 100 |


## Data Directory
```
\_data
    \_ train
        \_ 0.png , 1.png, 2.png ........
        \_ train_label_COVID.txt and train_label_COVID.xlsx
    \_ validate
        \_ 0.png , 1.png, 2.png ........
        \_ validate_label_COVID.txt and validate_label_COVID.xlsx
    \_ test
        \_ 0.png , 1.png, 2.png ........
        \_ test_label_COVID.txt and test_label_COVID.xlsx        

```

## Data Sample
<img width=200 src="sample_image/Negative_1.png"/> 음성_1                                  
<img width=200 src="sample_image/Positive_1.png"/> 양성_1     
<img width=200 src="sample_image/Negative_2.png"/> 음성_2                                  
<img width=200 src="sample_image/Positive_2.png"/> 양성_2     


## Label Sample
```
# train_label_COVID.txt
0.png 0
1.png 1
2.png 0
3.png 1
4.png 1
5.png 1
...
# train_label_COVID.xlsx
0.png	0
1.png	1
2.png	0
3.png	1
4.png	1


## Metric
평가를 위한 Metric : Accuracy

```

## Description
```
환자의 흉부 CT 이미지를 입력으로, 환자의 COVID 양성/음성 분류
```

## Commands
```
# train
python main.py 

# test (for submission)
python main.py --batch=4 --model_name="1.pth" --prediction_dir="prediction" --mode="test" 


All options in example commands are default value.
옵션은 main.py 파일 참고
```
