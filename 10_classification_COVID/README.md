# Classification_COVID

## Task
환자의 흉부 CT 이미지를 입력으로, 환자의 COVID 양성/음성 이진 분류 \
Input : 흉부 CT 이미지(384x384x3) \
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
        \_ 0.png , 1.png, 2.png ..., train_label_COVID.txt and train_label_COVID.xlsx
    \_ validate
        \_ 0.png , 1.png, 2.png ..., validate_label_COVID.txt and validate_label_COVID.xlsx
    \_ test
        \_ 0.png , 1.png, 2.png ..., test_label_COVID.txt and test_label_COVID.xlsx        

```

## Data Sample
<img width=200 src="sample_image/Negative_1.png"/> 음성_1 (384x384x3)                                 
<img width=200 src="sample_image/Positive_1.png"/> 양성_1 (384x384x3)    
<img width=200 src="sample_image/Negative_2.png"/> 음성_2 (384x384x3)                                
<img width=200 src="sample_image/Positive_2.png"/> 양성_2 (384x384x3)   


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

```
## Metric
```
평가를 위한 Metric : Accuracy
```

## Commands
```
# train
python main.py 

# test (for submission)
python main.py --batch=4 --mode="test" --model_name 1.pth

모든 옵션은 default value가 있음
옵션은 main.py 파일 참고
```
