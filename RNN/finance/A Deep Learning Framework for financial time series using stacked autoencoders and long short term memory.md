# Introduction

이 글은  PLOS에 게제된 __WeI Bao, Jun Yue, Yulei Rao__의 A Deep Learning Framework for financial time series using stacked autoencoders and long short term memory의 글의 정리 글입니다.


# 요약

특이점은 __SAE(Stacked Auto Encoder)__와 __LSTM__을 사용 하였다는 점입니다. 저자는 주가 시장을 예측 하기 위해서 RNN의 계열인 LSTM을 사용 하는데, 주목 할 점은 시계열 데이터인 주가 데이터는 변동이 자주 있는 데이터 이다 보니 데이터에 노이즈가 많이 낄 수 밖에 없는데  __Wave Transformation__으로 노이즈를 제거 하였습니다. 
