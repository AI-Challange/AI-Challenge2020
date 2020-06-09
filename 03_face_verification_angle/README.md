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
# train_label.txt
17080801_S001_L10_E01_C10_cropped.jpg 17080801_S001_L2_E01_C10_cropped.jpg

(front_image_name) (side_image_name)
```

## Metric
```
F1 score
```

## Description
```
For given image, baseline model just do convolutions and decovolutions generating new image that has original image size
```

## Commands
```
# train
python main.py --lr=0.001 --cuda=True --num_epochs=10 --print_iter=10 --model_name="model.pth" --prediction_dir="prediction" --batch=4 --mode="train"

# test (for submission)
python main.py --batch=4 --model_name="1.pth" --prediction_dir="prediction" --mode="test" 


All options in example commands are default value.
If you have problem with os.mkdir or shutil.rmtree in baseline code, manually remove prediction_dir and create prediction_dir
```
