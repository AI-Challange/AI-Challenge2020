# Improved Illumination

## Task
Input : 흉부 CT 이미지
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
        \_ train_label_COVID.txt or train_label_COVID.xlsx
    \_ validate
        \_ 0.png , 1.png, 2.png ........
        \_ validate_label_COVID.txt or validate_label_COVID.xlsx
    \_ test
        \_ 0.png , 1.png, 2.png ........
        \_ test_label_COVID.txt or test_label_COVID.xlsx        

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
```
The average of L1 Losses 

"bright_image_name" always has illumination code "L2"

Images that have "L2" are our answer images

So we are going to evaluate using them
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
