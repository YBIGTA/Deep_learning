# Introduction

이 글은  PLOS에 게제된 **WeI Bao, Jun Yue, Yulei Rao** 의 A Deep Learning Framework for financial time series using stacked autoencoders and long short term memory의 글의 정리 글입니다.


# 요약

특이점은 **SAE(Stacked Auto Encoder)** 와 **LSTM**을 사용 하였다는 점입니다. 일단 시계열 데이터인 주가 데이터는 변동이 자주 있는 데이터 이다 보니 데이터에 노이즈가 많이 낄 수 밖에 없는데  **Wave Transformation**으로 노이즈를 제거 하였습니다.
