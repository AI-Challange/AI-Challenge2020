# Speech To Text

## Task
STT with Noise : 잡음이 함께 녹음된 음성을 text로 변환\
Input : 성인(잡음 포함)의 단어/문장 발화 녹음 파일 (※ 잡음 : 정지 및 주행중인 자동차 환경, TV 혹은 라디오 소리)\
Output : 발화 Text

## Dataset
| Phase | # |
| - | - |
| train | 7,886 |
| validate | 1,000 |
| test | 1,938 |


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
./1.pcm 동문빌라
./2.pcm 롯데마트화성점
./3.pcm 방은바다가보이는방향으로예약해
./4.pcm 암사선사유적전시관
./5.pcm 남기환
./6.pcm 서강대교
./7.pcm 부산역
./8.pcm 부가기능
./9.pcm 에이엔씨약국
./10.pcm 하남하이플라자
...
./317.pcm 을지로삼가이호선지하철역
...

※ 발화 레이블에는 띄어쓰기 및 특수문자(., ?, ! 등) 없음
※ 영어(알파벳), 숫자 발음은 모두 한국어로 전사 
   예시> ./9.pcm : (ANC -> 에이엔씨), ./317.pcm : (을지로3가2호선 -> 을지로삼가이호선)

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
python main.py --batch=4 --model_name="1.pth" --mode="test" --model_name 1.pth

모든 옵션은 default value가 있음
옵션은 main.py 파일 참고
```
