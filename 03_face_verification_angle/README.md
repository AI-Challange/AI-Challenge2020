# Face Verification(angle)

## Task
```
측면 얼굴 이미지로 정면 얼굴 이미지와의 동일인 여부를 판단하는 문제입니다.
```

## Dataset
<img width=900 src="images_for_desc/image.png"/>

| Phase | front | side | total |
| - | - | - | - |
| train | 22,050 | 44,100 | 66,150 |

```
train data로 정면과 측면 얼굴 이미지를 다양하게 제공합니다. 
학습 시간 등을 고려하여 dataloader로 부터 자유롭게 로드하여 사용하시면 됩니다.
```

| Phase | True | False | total |
| - | - | - | - |
| validate | 5,000 | 5,000 | 10,000 |
| test | 10,000 | 10,000 | 20,000 |

```
validate와 test data는 정면과 측면 이미지 한 쌍을 Pair로 제공합니다.
```

## Data Directory
```
\_data
    \_ train
        \_ ??????_S001_L??_E??_C??_cropped.jpg (images)
        \_ train_meta.csv
        \_ train_label.csv (train 시 dataloader 코드에서 생성되어 저장됨)
    \_ validate
        \_ ??????_S001_L??_E??_C??_cropped.jpg (images)
        \_ validate_label.csv
    \_ test
        \_ ??????_S001_L??_E??_C??_cropped.jpg (images)
        \_ test_label.csv (dummy labels)

image_name = "Person"_S001_"Illumination"_"Expression"_"Camera_Angle"_cropped.jpg
```

## Data Sample
<img width=400 src="images_for_desc/image2.png"/>　　　<img width=400 src="images_for_desc/image3.png"/>

## Label
```
# train_meta.txt
(image_name)                         (face_id)  (ang_option)
19082212_S001_L1_E01_C8_cropped.jpg  19082212   front

# *****_label.txt
(none_image_name)                    (acc_image_name)                    (label)
19082212_S001_L1_E01_C8_cropped.jpg  19082212_S001_L5_E02_C3_cropped.jpg  0

```

## Metric
```
F1 score
```

## Description
```
측면 얼굴(side) 이미지로 정면 얼굴(front) 이미지와의 동일인 여부를 판단하는 문제입니다.
정면 1장과 측면 1장이 한 pair로 주어집니다.(이 때, 조명 밝기와 표정은 랜덤으로 주어짐)
label은 동일인일 때 0, 동일인이 아닐 때 1로 예측하시면 됩니다.

Baseline code는 siamnetwork로 모델링되어 있으며 이는 자유롭게 변경하여 작성하시면 됩니다.
train data 기준으로 만들 수 있는 최대 Pair의 수는 약 10억 쌍 정도이나 
모든 쌍을 학습에 사용하지 않고 train 실행 시 학습에 사용할 pair를 만들도록 작성되어 있습니다.
이 부분 역시 자유롭게 변경하여 사용하실 수 있습니다.
다만, 정확한 evaluation을 위해 label file 양식은 변경하실 수 없습니다.
```

## Commands
```
# train
python main.py --lr=0.001 --cuda=True --num_epochs=10 --print_iter=10 --prediction_dir="prediction" --batch=16 --mode="train"

# test (for submission)
python main.py --batch=16 --model_name="1.pth" --prediction_dir="prediction" --mode="test" 


All options in example commands are default value.
```
