# Speech To Text

## Task
STT for Child : 어린이(초등 1~4학년)의 음성을 Text로 변환 \
Input : 어린이의 단어/문장 발화 녹음 파일 \
Output : 발화 Text

## Dataset
| Phase | # |
| - | - |
| train | 11,340 |
| validate | 2,430 |
| test | 2,430 |


## Data Directory
```
\_data
    \_ train
        \_ xxxxxxxxxxx.pcm ..., train_label.txt
    \_ validate
        \_ xxxxxxxxxxx.pcm ..., validate_label.txt
    \_ test
        \_ xxxxxxxxxxx.pcm ..., test_label.txt

```
## Data Sample
```
sample_pcm 폴더 확인
 - 예시 pcm 데이터 제공
 - 예시 pcm 데이터를 wav 파일로 변환하여 제공
※ Meta Data for .pcm : channels=1, bit_depth=16, sampling_rate=16000
```

## Label Sample
```
# train_label.txt
./I0007_M1_OFC_105.pcm 사자
./M0193_M4_OFC_063.pcm 원숭이
./M0007_M1_OFC_043.pcm 사과
./I0129_M3_OFC_075.pcm 컴퓨터
./I0187_F4_OFC_084.pcm 저에게친절하게대해주시기때문입니다
./M0192_M4_OFC_063.pcm 주사
./M0066_M2_OFC_083.pcm 아빠
./I0010_M1_OFC_007.pcm 돌
./G0012_M1_OFC_052.pcm 선생님
./M0187_F4_OFC_036.pcm 세탁기
...

※ 발화 레이블에는 띄어쓰기 및 특수문자(., ?, ! 등) 없음

```

## Metric
```
평가를 위한 Metric : 음절단위 F1

```

## Commands
```
# train
python main.py 

# test (for submission)
python main.py --batch=4 --model_name="1.pth" --prediction_dir="prediction" --mode="test" 

모든 옵션은 default value가 있음
옵션은 main.py 파일 참고
```
