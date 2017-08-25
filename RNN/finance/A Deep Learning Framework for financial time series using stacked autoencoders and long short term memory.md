# Introduction

이 글은  PLOS에 게제된 **WeI Bao, Jun Yue, Yulei Rao** 의 A Deep Learning Framework for financial time series using stacked autoencoders and long short term memory의 글의 정리 글입니다.


# 요약

특이점은 **SAE(Stacked Auto Encoder)** 와 **LSTM**을 사용 하였다는 점입니다. 일단 시계열 데이터인 주가 데이터는 변동이 자주 있는 데이터 이다 보니 데이터에 노이즈가 많이 낄 수 밖에 없는데  **Wave Transformation**으로 노이즈를 제거 하였습니다. 이렇게 노이즈가 제거된 데이터를 SAE를 통해서 feature를 생성 합니다.  SAE를 통해 만들어진 high level의 feature를 이용하여 LSTN에 input data로 사용하여 다음날의 종가를 예측하는식으로 model을 트레이닝 합니다. 기존의 RNN, LSTN,  WT-LSTN과 MAPE(Mean Average Percentage Error)를 이용한 accuracy와 subsection prediction method를 적용한 profitability 모두에서 제일 나은 성과를 얻었습니다.
