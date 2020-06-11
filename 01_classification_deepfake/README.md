# Deepfake Classification

## Task
주어진 안면 이미지의 진짜, 가짜 여부를 분류하는 문제\
Input : 안면 이미지\
Output : 진짜, 가짜 여부(real: 0, fake: 1)

## Dataset
### image resolution 112x112x3
| Phase | # |
| - | - |
| train | 240000 |
| validate | 80000 |
| test | 80000 |


## Data Directory
```
\_data
    \_ train
        \_ 0.png , 1.png, 2.png ..., train_label.txt
    \_ validate
        \_ 0.png , 1.png, 2.png ..., validate_label.txt
    \_ test
        \_ 0.png , 1.png, 2.png ..., test_label.txt

```

## Data Sample
<img width=600 src="images_for_desc/deppfake-dataset.png"/> 

## Label Sample
```
# train_label.txt
69a1d6e71f594437b0dd71b4d59fbd38.jpg 1
a96ea3b505f7439999707826459c5501.jpg 1
57a63e707e2d4ea5a5d3f5ef4ea1e98a.jpg 0
45fcb1504c49407183e4f3667b06dee5.jpg 1
802f279143f448898966b8c606c15008.jpg 1
5c5093d5ccb9412493d7eadd66589c13.jpg 0
e9d9d8754b304191bee5fd097def2022.jpg 1
ccfd3fb6812847389d42888f54a604a3.jpg 1
11010d96d13d4532bc63598b1275335c.jpg 1
87c70f84c0f744ffaa732b2eb6b55701.jpg 0
a87e6eed1ce64e5aa867166fe6741c07.jpg 0
bda9bc38139a4a6c9489ee1f9871d63b.jpg 1
```

## Metric
```
평가를 위한 Metric : F1 score
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

## Submission
```
예측 결과 파일을 제출하시면 됩니다.

예측 결과 파일은 image_name label 형태입니다.(prediction.txt 참조)
label 부분을 예측 값으로 작성하시어 제출하시면 됩니다.
```
