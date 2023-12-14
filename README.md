# hotelreviewkor

# <div align=center> KOELECTRA를 활용한 호텔리뷰 평점 감성 분석 </div>
<img src = https://github.com/seonggegun/hotelreview/assets/79897862/3a2b34b4-1ba6-4325-a443-14907cbacdb2><br>
[출처: bing create](https://www.bing.com/create)

# 1. 개요
이 프로젝트의 목적은 KOELECTRA를 활용하여 호텔 리뷰의 긍부정을 예측하고자 한다.<br> 
호텔 리뷰는 호텔 예약 고객에게 중요한 정보를 제공하는 주요 소스로 사용된다. 이러한 리뷰를 통해 호텔의 품질, 서비스 및 시설에 대한 정보를 얻어 만족할만 예약이 가능할것이다.

## 1.1 문제 정의
호텔 리뷰는 호텔 예약에 있어 가장 보편적으로 활용되는 정보의 창구이다.
호텔 리뷰는 고객들이 자신의 경험과 의견을 공유하고, 잠재적인 호텔 예약 고객에게 중요한 결정을 돕는 중요한 정보원이며 이러한 리뷰는 호텔의 서비스 품질, 시설, 위치, 청결도, 가격, 음식 등 다양한 측면을 다루며, 고객들이 호텔을 선택하거나 피하는 데 큰 영향을 미친다. 더 나아가, 호텔 리뷰는 호텔 업계에서 품질 향상을 위한 중요한 피드백 제공자로서의 역할을 한다.<br>
[호텔 리뷰가 호텔 성행 상관관계를 나타내는 자료](http://kmr.kasba.or.kr/xml/25550/25550.pdf)
이 자료에 따르면 호텔의 잠재 고객들은 이미 호텔 상품을 소비해본 사람들이 게재하는 온라인 리뷰를 통해 더 정확하고 신뢰할만한 정보를 탐색하는 경향이 있다. 따라서, 잠재 고객들이 유용성을 인식한 온라인 리뷰는 그들의 구매결정에 영향을 미치는 중요한 역할을 하는 것으로 나타났다 .
## 1.2 데이터 및 모델 개요
데이터는 [깃허브](https://github.com/Dabinnny/Review_SentimentAnalysis/tree/main/Data) 저장소에서 공개된 크롤링한 호텔리뷰 데이터를 사용하여 한국어 자연어 처리 모델인 KOELECTRA를 활용하여 호텔 리뷰의 긍부정 예측을 하고자 한다.<br><br>

|Unnamed: 0|hotel|star|reviews|date|real_date|length|
| --- | --- | --- | --- | --- | --- | --- |
| 번호 | 호텔명 | 별점 | 리뷰 | 날짜 | 실시간 | 문장길이 |


<img src = https://github.com/seonggegun/hotelreview/assets/79897862/2ece0c92-ec90-4da3-9117-bd7ec7a21da0> <br>
<table>
  <tr><th></th><th>별점</th><th>건수</th></tr>
  <tr><th rowspan='2'>긍정 (21,281)</th><th>5</th><td>17,723</td></tr>
  <tr><th>4</th><td>3,558</td></tr>
  <tr><th rowspan='3'>부정 (1637)</th><th>3</th><td>1,045</td></tr>
  <tr><th>2</th><td>291</td></tr>
  <tr><th>1</th><td>301</td></tr>
  <tr><th rowspan='1'>제외 (3)</th><th>0</th><td>3(블라인드조치된 언행)</td></tr>
  <tr><th colspan='2'>계</th><td>22921</td></tr>
</table> <br>
긍정 21,281 부정 1,637건으로 긍정리뷰가 압도적으로 많음을 알수있다.

# 2. 데이터
## 2.1 데이터 소스
[호텔리뷰데이터](https://github.com/Dabinnny/Review_SentimentAnalysis/tree/main/Data)
|Unnamed: 0|hotel|star|reviews|date|real_date|length|
| --- | --- | --- | --- | --- | --- | --- |
|0|그랜드 하얏트 서울|5| 최고의 주말 데이트를 했습니다| 1일전 | 2022. 03. 14 |16|
|1|그랜드 하얏트 서울|5| 조식 맛집👍 | 2일전 | 2022. 03. 13 |6|
|2|그랜드 하얏트 서울|5| 야경이 너무너무 멋졌어요^^ | 2022. 03. 07 | 2022. 03. 07 |15|
|3|그랜드 하얏트 서울|4| 체크인 시간 전에 도착 시 대기 할 공간이 미흡했던 것같고, 시설 이용할 수 있는게 한정적이여서 아쉬움이 컸음 | 2022. 03. 07 | 2022. 03. 07 |61|
| ... | ...| ... | ... | ... | ... | ... |
|22920|파라다이스호텔&리조트,파라다이스시티|5| 너무 행복했어용 | 2019. 04. 12 | 2019. 04. 12 |8|
|22921|파라다이스호텔&리조트,파라다이스시티|5| 정말 좋네요!! | 2019. 04. 07 | 2019. 04. 07 |8|
|22922|파라다이스호텔&리조트,파라다이스시티|5| 말이 필요 없는 최고... 네스트랑 비교 불가 | 2019. 03. 27 |2019. 03. 27 |25|<br>
## 2.2 탐색적 데이터 분석


## 2.3 데이터 전처리


|reviews|'star'|'label'|
| --- | --- | --- |
| 호텔리뷰 | 평점 | 긍정0,부정1 |
| 최고의 주말 데이트를 했습니다 | 5 | 0 |
| 야경이 너무너무 멋졌어요^^ | 5 | 0 |
| 친절하고 시설도 좋네요 | 5 | 0 |
| 체크인 시간 전에 도착 시 대기 할 공간이 미흡했던 것같고, 시설 이용할 수 있는게 한정적이여서 아쉬움이 컸음 | 4 | 0 |
| 나쁘잔않지만 가성비를따지자면 그닥.. | 3 | 1 |
| 서비스가 다 따로노는 느낌이네요..기대하고 간곳인데 기대이하네요 | 3 | 1 |
| 배드 밑에 뭘 깔아놓으셨는지 울퉁불퉁해서 허리아파서 잠을 못잤네요 | 1 | 0 |
| 개인 위생용품 자체가 없고, 냉방과 난방 둘 중 하나만 선택 가능합니다. | 2 | 0 | <br>
<img src = https://github.com/seonggegun/hotelreview/assets/79897862/c815c74c-11d9-4540-b41d-501969e5fc01 width="400" height="200">
<img src = https://github.com/seonggegun/hotelreview/assets/79897862/d59b1da7-9648-449b-ac13-8a2374564f4e width="400" height="200">


1. 엑셀내에서 쓸모없는 Unnamed, hotel(호텔이름), date(리뷰남긴날짜),real_date(실제 날짜)를 제거
2. 평점 1&tilde;3점은 부정(1), 4&tilde;5점은 긍정(0)으로 변환
3. 긍정데이터 부정데이터 비율맞추기
4. 각 전처리 후, 결측치 확인<br>

```
def del_percent():
    has_missing_values = df.isnull().values.any()
print("결측치 확인 =", has_missing_values)
# 결측치가 있으면 해당 행 제거
if has_missing_values:
    df = df.dropna(how='any')
# 처리된 데이터 개수 출력
print("처리된 데이터 개수 =", len(df))
```
<br>
<img src= https://github.com/seonggegun/hotelreview/assets/79897862/18557f41-718a-46a7-a1af-797b667809b9 width="300" height="100"> <br>
2000개 데이터에 결측치는 없음을 알수있다.<br>
5. 특수 이모지 및 전처리 <br>
<img src= https://github.com/seonggegun/hotelreview/assets/79897862/b6bb2e43-d6ac-4cd6-ae00-69d3d2dce665><br>
좌쪽이 전처리 전 우쪽이 전처리 후 로 이모지가 제거됨을 알수있다.

```
# 모델 정의
model = ElectraForSequenceClassification.from_pretrained('koelectra-small-v3-discriminator', num_labels=2)

# Optimizer 및 Scheduler 설정
optimizer = torch.optim.Adam(model.parameters(), lr=3e-04, eps=1e-06, betas=(0.9, 0.999))
epoch = 3
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)*epoch)

# 학습 및 검증 루프
for e in range(0, epoch):
    print(f'\n\nEpoch {e+1} of {epoch}')
    print(f'** 학습 **')
    t0 = time.time()
    total_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        # 중략
        
    # 평가 루프
    print('\n\n** 검증 **')
    t0 = time.time()
    model.eval()
    eval_loss, eval_accuracy, eval_steps, eval_example = 0, 0, 0, 0
    for batch in validation_dataloader:
```

# 3. 재학습 결과
<img src = https://github.com/seonggegun/hotelreview/assets/79897862/6c66f7c2-e2d4-4b69-8748-c811b36294dd>

## 3.1 개발 환경
- <img src="https://img.shields.io/badge/PyCharm -000000?style=for-the-badge&logo=PyCharm&logoColor=white">      <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white">
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=red">   <img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=Numpy&logoColor=turquoise">
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=Pandas&logoColor=Dark Purple">      <img src="https://img.shields.io/badge/Transformers-007acc.svg"> <img src="https://img.shields.io/badge/Scikit-learn-F7931E?style=for-the-badge&logo=Scikit-learn&logoColor=orange">
## 3.2 KOELECTRA fine-tuning
22000건중 긍정이 너무 압도적으로 많아 부정 1000건 긍정 1000건만 추출하여 진행했다.
## 3.3 학습 결과 그래프

# 4. 배운점
