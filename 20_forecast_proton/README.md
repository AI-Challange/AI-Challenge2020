# Classification_COVID

## Task
```
태양관측위성으로 수집되는 데이터를 바탕으로 태양입자유입(Proton Flux) 양상 예측 문제

Input : SWE, EPM, X-ray 
Output : Proton
```
## Description
```
- 태양의 플레어, 코로나 등에 의해 고에너지 입자가 방출되며, 1시간 ~ 수일 이내에 지구에 도달
- 이 때, X-ray, SWE, EPM 등의 정보를 양성자(Proton Flux)보다 먼저 관측 할 수 있음
- 지구에 피해를 줄 우려가 있는 요소는 양성자(Proton Flux)임
- 양성자의 Traval Time은 유동적임 
- 각 관측 데이터는 수집 주기(Time Step)가 다름
- Proton :5분 / x-ray : 1분 / SWE : 약 1분 / EPM : 약 5분 
- SWE와 EPM은 수집주기가 다소 변동적임
- Input Data(X-ray, SWE, EPM) 중 -100은 위성의 고장으로 인해 해당 데이터가 관측이 되지 않았다는 메세지입니다. 
  (Baseline에서는 임의로 0으로 변환합니다. 해당 Empty Cell의 데이터를 추정, 예측하여 Output에 반영한느 것 또한 참가자의 능력으로 평가합니다.)
- Output Data(Proton) 중 -100은 위성의 고장으로 인해 해당 데이터가 관측이 되지 않앗다는 메시지니다.  해당 Step은 평가에서 제외됩니다.

태양으로부터 관측되는 데이터는 매 사건마다 Traval Time이 다를 수 있어 관측되기 까지의 시간/ 혹은 영향을 주는 기간이 다를 수 있으며, 
각 데이터의 관측 주기도 다르니 참고바랍니다.
```


## Dataset
| Phase | 기간 | Solar Proton Events |
| - | - | - |
| train | 1998/02/04 ~ 2005/09/10 | 88 |
| validate | 2005/09/11 ~ 2012/07/11 | 19 |
| test | 2012/07/12 ~ 2017/12/31 | 20 |

※ 참고 : https://umbra.nascom.nasa.gov/SEP/ (Solar Proton Events Affecting the Earth Environment)


## Data Directory
```
\_data
    \_ train
        \_ train_AC_H0_SWE.csv, train_AC_H1_EPM.csv, train_proton.csv, train_xray.csv
    \_ validate
        \_ val_AC_H0_SWE.csv, val_AC_H1_EPM.csv, val_proton.csv, val_xray.csv
    \_ test
        \_ test_AC_H0_SWE.csv, test_AC_H1_EPM.csv, test_proton.csv, test_xray.csv

```

## Data Sample
### SWE
```
time_tag	                H_DENSITY_#/cc	SW_H_SPEED_km/s
2005-09-11T00:00:53.000Z	    -100	        -100
2005-09-11T00:01:57.000Z	    -100	        -100
                            .
                            .
2007-02-14T19:03:06.000Z	    2.8289	        674.1
2007-02-14T19:04:10.000Z	    3.1459	        669.26
```

### EPM
```
time_tag	P1P_.047-.066MEV_IONS_1/(cm**2-s-sr-MeV) ... P8P_1.89-4.75MEV_IONS_1/(cm**2-s-sr-MeV)
2005-09-11T00:01:00.000Z	152870	48966	33411	23198	17091	12280	7512.8	2836
2005-09-11T00:06:00.000Z	161880	56622	39165	26999	19946	13861	8183.1	2946.3
2005-09-11T00:11:00.000Z	156540	54534	38960	27481	20267	14088	8352.8	3000.3
                    .
                    .
```

### xray
```
time_tag	            xs	      xl
2005-09-11 00:00.0	   -100      -100
2005-09-11 01:00.0	   -100	     -100
                    .
                    .
2007-03-15 18:09        4.78E-09 3.73E-09
2007-03-15 18:10        4.78E-09 3.73E-09
```


## Label Sample
### Proton
```
time_tag    proton
00:00.0     -100
05:00.0     955
10:00.0     1010
15:00.0     1050
20:00.0     1000
25:00.0     1090
   .
   .
```

## Metric
```
평가를 위한 Metric : RMSE(Custom)
weight : proton Time Step 기준으로 전체 데이터수/해당 등급 데이터 수
```
<img width=400 src="RMSE(Custom).PNG"/>

| Proton Value | Weight |
| - | - |
| 0 ~ 10 | 1 | 
| 10 ~ 100 | 68 |
| 100 ~ 1000 | 182 |
| 1000 ~ 10000 | 809 |
| 10000 ~ | 6041 |

## Commands
```
# train
python main.py 

# test (for submission)
python main.py --model_name="1" --mode="test"

모든 옵션은 default value가 있음
옵션은 main.py 파일 참고
```

## Reference
본 문제에서는 베이스라인에서 특정 모델 아키텍쳐를 제공하지 않습니다.
(데이터들의 시관관계 맵핑  사용자의 능력으로 평가하기 위함합니다.)
RNN Model Architecture : https://github.com/chickenbestlover/RNN-Time-series-Anomaly-Detection

