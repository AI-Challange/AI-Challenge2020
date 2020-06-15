# 13_OCR_handwriting

## Task
```
주어진 이미지의 문장을 Text로 추출
```

## Dataset
* image size는 이미지마다 다름. Label Sample 참조

| Phase | Image | Label|
| - | - | - |
| train | 36,961 | 1 |
| validate | 7,920 | 1 |
| test | 7,919 | 1 |


## Data Directory
```
\_data
    \_ train
        \_ *.png (sentence_images), train.json (Label)
    \_ val
        \_ *.png (sentence_images), val.json (Label)
    \_ test
        \_ *.png (sentence_images), test.json (Label)
```

## Data Sample
<img width=350 src="images_for_desc/sample_1.png"/>　　　<img width=350 src="images_for_desc/sample_2.png"/>
<img width=350 src="images_for_desc/sample_3.png"/>　　　<img width=350 src="images_for_desc/sample_4.png"/>


## Label Sample
```
{
  "annotations": [
    {
      "width": 3740,
      "height": 176,
      "file_name": "00000010.png",
      "text": "근 노동현안에 대해 변화하는 만큼 믿고 지켜볼 것\"이라고 말했다. 한"
    },
    {
      "width": 3736,
      "height": 178,
      "file_name": "00000035.png",
      "text": "시장에 뛰어들었다. 한국야쿠르트는 소수의 고객에게만 판매하는 '내"
    },
           .
           .
           .
}
```


## Output Sample
```
{
  "predict": [
    {
      "image_path": "./data/val/00000010.png",
      "prediction": "Sample"
    },
    {
      "image_path": "./data/val/00000035.png",
      "prediction": "예시"
    },
        .
        .
        .
}
```


## Metric
```
각 문장별로 Error Rate를 검사하는 
mWER(Mean Word Error Rate)를 사용했습니다.

mWER = sum(WERs) / len(Inputs)
```
<img width=350 src="images_for_desc/wer.png"/>


## Description
```
dataloader.py : img name기준 오름차순으로 json 파일 안에 라벨 정보를 불러온 뒤 label 이라는 스트링 형태로 저장, 한 이미지 tensor와 그 이미지에 대한 label이 label이라는 스트링으로 매칭되는 방식
model.py : CRNN으로 구현됨
main.py : train, test 함수 구현, test의 경우 submission file 형식으로 저장됨.
evaluate.py : submission file을 통해 성능 평가
```


## Commands
```
# train
python main.py 

# test (for submission)
python main.py --model_name="1.pth" --prediction_dir="prediction" --mode="test" 

모든 옵션은 default value가 있음
옵션은 main.py 파일 참고
```
