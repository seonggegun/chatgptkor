# hotelreviewkor

# KOELECTRA를 활용한 호텔리뷰 평점 감성 분석
<img src =https://github.com/seonggegun/hotelreview/assets/79897862/ff360823-9718-4995-af40-081c6aaf80db.jpg width="2000" height="500">**출처:** [링크]([https://example.com](https://kr.freepik.com/free-vector/flat-hotel-review-background_4433293.htm)) 이미지출처


# 1. 개요
이 프로젝트의 목적은 KOELECTRA를 활용하여 호텔 리뷰의 감성을 분석하는 것이다.<br> 
호텔 리뷰는 호텔 예약 고객에게 중요한 정보를 제공하는 주요 소스로 사용된다. 이러한 리뷰를 통해 호텔의 품질, 서비스 및 시설에 대한 정보를 얻어 만족할만 예약이 가능할것이다.

## 1.1 문제 정의
호텔 리뷰는 호텔 예약에 있어 가장 보편적으로 활용되는 정보의 창구이다.
[호텔 리뷰가 호텔 성행 상관관계를 나타내는 자료출처:](http://kmr.kasba.or.kr/xml/25550/25550.pdf)
호텔 리뷰는 고객들이 자신의 경험과 의견을 공유하고, 잠재적인 호텔 예약 고객에게 중요한 결정을 돕는 중요한 정보원이며 이러한 리뷰는 호텔의 서비스 품질, 시설, 위치, 청결도, 가격, 음식 등 다양한 측면을 다루며, 고객들이 호텔을 선택하거나 피하는 데 큰 영향을 미친다. 더 나아가, 호텔 리뷰는 호텔 업계에서 품질 향상을 위한 중요한 피드백 제공자로서의 역할을 한다.

## 1.2 데이터 및 모델 개요
데이터는 [이곳](https://github.com/Dabinnny/Review_SentimentAnalysis/tree/main/Data) 에서 크롤링한 호텔리뷰 데이터를 사용하여 한국어 자연어 처리 모델인 KOELECTRA를 활용하여 호텔 리뷰의 긍부정 예측을 하고자 한다.<br><br>

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


<img src = https://github.com/seonggegun/hotelreview/assets/79897862/2ece0c92-ec90-4da3-9117-bd7ec7a21da0> <br>
리뷰데이터22921개(호텔명, 평점, 리뷰텍스트, 리뷰작성일)


# 2. 데이터
## 2.1 데이터 소스
[호텔리뷰데이터](https://github.com/Dabinnny/Review_SentimentAnalysis/tree/main/Data)
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
<img src = https://github.com/seonggegun/hotelreview/assets/79897862/7938e494-b162-4d8d-a8db-48c3cec559c0>

1. 엑셀내에서 쓸모없는 Unnamed, hotel(호텔이름), date(리뷰남긴날짜),real_date(실제 날짜)를 제거
2. 평점 1&tilde;3점은 부정(1), 4&tilde;5점은 긍정(0)으로 변환
3. 각 전처리 후, 18글자 이하의 리뷰 갯수 비율 확인하는 함수<br>
<img src= https://github.com/seonggegun/hotelreview/assets/79897862/1ffed558-2136-414c-b46d-0a27f2c398e1 width="500" height="100"> <br>
<img src= https://github.com/seonggegun/hotelreview/assets/79897862/c9d559ae-df9e-4ce0-bddc-5eee8615a1fa width="300" height="100"> <br>
18글자 이하 리뷰갯수: 13466, 비율 :58.749618254002876 임을 알수있다.<br>
4. 특수 이모지 및 전처리 <br>
<img src= https://github.com/seonggegun/hotelreview/assets/79897862/14fd5ea0-ce0c-43cd-abd2-cd84a0f06dec width="500" height="300">
<img src= https://github.com/seonggegun/hotelreview/assets/79897862/30f99945-c79a-40e4-ab4a-3daf393b98a6 width="500" height="300">



# 3. 재학습 결과
## 3.1 개발 환경
- pycharm, python, torch, pandas, ...
-
## 3.2 KOELECTRA fine-tuning
## 3.3 학습 결과 그래프

# 4. 배운점
