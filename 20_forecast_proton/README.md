# Classification_COVID

## Task
ㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇ \
Input : SWE, EPM, X-ray \
Output : Proton

## Dataset
| Phase | 기간 | Solar Proton Events |
| - | - | - |
| train | 1998/01/01 ~ 2005/09/09 | 88 |
| validate | 2005/09/10 ~ 2012/07/10 | 19 |
| test | 2012/07/11 ~ 2017/12/31 | 20 |

※ 참고 : https://umbra.nascom.nasa.gov/SEP/ (Solar Proton Events Affecting the Earth Environment)


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

이미지로 예시

## Label Sample


이미지로 예시


## Metric
```
평가를 위한 Metric : RMSE(Custom) * weight (각 Time step의 proton value에 비례)
```
<img width=300 src="RMSE_Custom.png"/>

## Commands
```
# train
python main.py 

# test (for submission)
python main.py --batch=4 --model_name="1.pth" --mode="test" --model_name 1.pth

모든 옵션은 default value가 있음
옵션은 main.py 파일 참고
```
