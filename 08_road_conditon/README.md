# Surface segmentation

## Task
```
Image segmentation
```

## Dataset
| train | image: 32,467 | mask_image: 32,467 | xml_file: 540 |

| validate | image: 7,028 | mask_image: 7,028 | xml_file: 124 |

| test | image: 6,904 | mask_image: 6,857 | xml_file: 136 |


## Data Directory
```
\_data
    \_ train
        \_ Surface_***
        \_ MASK, *.xml , *.jpg (images)
            \_ *.png (mask_images)
    \_ val
        \_ MASK, *.xml , *.jpg (images)
            \_ *.png (mask_images)
    \_ test
        \_ MASK, *.xml , *.jpg (images)
            \_ *.png (mask_images)

image_name = "MP_SEL_SUR_033182.jpg"
mask_image_name = "MP_SEL_SUR_033182.png"
```

## Data Sample
<img width=350 src="sample_images/MP_SEL_SUR_000001.jpg"/>　　　<img width=350 src="sample_images/MP_SEL_SUR_000001.png"/>


## Label
```
# *.xml
<img width=500 src="sample_images/xml.PNG"/>　　　


```

## Metric
```
mAP(mask IoU = 0.5)

we evaluate mask IOU and class so you must get mask array and predict class.
```

## Description
```
For given image, baseline model just do convolutions and decovolutions generating new image that has original image size
```

## Commands
```
# train
python main.py --num_classes=38 --lr=0.001 --cuda=True --num_epochs=10 --print_iter=10 --model_name="model.pth" --prediction_dir="prediction" --batch=4 --mode="train"

# test (for submission)
python main.py --batch=4 --model_name="1.pth" --prediction_dir="prediction" --mode="test" 


All options in example commands are default value.
If you have problem with os.mkdir or shutil.rmtree in baseline code, manually remove prediction_dir and create prediction_dir
```
