# Introduction

이 글은  PLOS에 게제된 **WeI Bao, Jun Yue, Yulei Rao** 의 A Deep Learning Framework for financial time series using stacked autoencoders and long short term memory의 글의 정리 글입니다.


# 요약

특이점은 **SAE(Stacked Auto Encoder)** 와 **LSTM**을 사용 하였다는 점입니다. 일단 시계열 데이터인 주가 데이터는 변동이 자주 있는 데이터 이다 보니 데이터에 노이즈가 많이 낄 수 밖에 없는데  **Wave Transformation**으로 노이즈를 제거 하였습니다. 이렇게 노이즈가 제거된 데이터를 SAE를 통해서 feature를 생성 합니다.  SAE를 통해 만들어진 high level의 feature를 이용하여 LSTN에 input data로 사용하여 다음날의 종가를 예측하는식으로 model을 트레이닝 합니다. 기존의 RNN, LSTN,  WT-LSTN과 MAPE(Mean Average Percentage Error)를 이용한 accuracy와 subsection prediction method를 적용한 profitability 모두에서 제일 나은 성과를 얻었습니다.

주가 시장 예측은 EMH(Efficient of Market Hypothesis)를 따르기에, 주가 시장의 효율성이 예측에 영향을 미칠 수 밖에 없습니다. 이러한 문제를 해결 하기 위해서 중국, 인도의 index를 발전 하고 있는 시장으로 정의하고, 미국의 index를 발전 된 시장으로 정의 하여 각각에 모델을 검증 함으로서 모델의 유용성을 확인 했습니다. 

Input으로 사용된 feature들은 3가지 historical variable 입니다. 1번째로는 **주가 거래 history** 입니다. 시작가, 최고가, 최저가, 종가(OHLC) 입니다. 2번째로는 주가 데이터의 **기술적인 indicator**입니다. 이 두가지는 일반적으로 많이 사용되는 variable입니다. 이 논문에서는 주가적으로 **macroeconmic(거시경제적)** 데이터도 variable로 사용했습니다. 이들은 거시경제적 데이터를 variable로 사용함으로서 deep learning이 invariant하고 abstract한 feature들을 만들거라고 기대하고 사용 하였습니다.

