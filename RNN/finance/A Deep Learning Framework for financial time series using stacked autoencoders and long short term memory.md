# Introduction

이 글은  PLOS에 게제된 **WeI Bao, Jun Yue, Yulei Rao** 의 A Deep Learning Framework for financial time series using stacked autoencoders and long short term memory의 글의 정리 글입니다.


# 요약

특이점은 **SAE(Stacked Auto Encoder)** 와 **LSTM**을 사용 하였다는 점입니다. 일단 시계열 데이터인 주가 데이터는 변동이 자주 있는 데이터 이다 보니 데이터에 노이즈가 많이 낄 수 밖에 없는데  **Wave Transformation**으로 노이즈를 제거 하였습니다. 이렇게 노이즈가 제거된 데이터를 SAE를 통해서 feature를 생성 합니다.  SAE를 통해 만들어진 high level의 feature를 이용하여 LSTN에 input data로 사용하여 다음날의 종가를 예측하는식으로 model을 트레이닝 합니다. 기존의 RNN, LSTN,  WT-LSTN과 MAPE(Mean Average Percentage Error)를 이용한 accuracy와 subsection prediction method를 적용한 profitability 모두에서 제일 나은 성과를 얻었습니다.

주가 시장 예측은 EMH(Efficient of Market Hypothesis)를 따르기에, 주가 시장의 효율성이 예측에 영향을 미칠 수 밖에 없습니다. 이러한 문제를 해결 하기 위해서 중국, 인도의 index를 발전 하고 있는 시장으로 정의하고, 미국의 index를 발전 된 시장으로 정의 하여 각각에 모델을 검증 함으로서 모델의 유용성을 확인 했습니다. 

Input으로 사용된 variable들은 3가지 historical variable 입니다. 1번째로는 **주가 거래 history** 입니다. 시작가, 최고가, 최저가, 종가(OHLC) 입니다. 2번째로는 주가 데이터의 **기술적인 indicator**입니다. 이 두가지는 일반적으로 많이 사용되는 variable입니다. 이 논문에서는 주가적으로 **macroeconmic(거시경제적)** 데이터도 variable로 사용했습니다. 이들은 거시경제적 데이터를 variable로 사용함으로서 **deep learning이 invariant하고 abstract한 feature들을 만들거라고 기대**하고 사용 하였습니다.

모델의 성능의 2가지 면에서 측정 되었습니다. 정확성과 수익성입니다. 정확성은 **MAPE(Mean Absolute Percentage Error)** 와 **Correlation(R)** 그리고 **Theil's inequaility coefficient(Theil U)** 로 측정되었습니다. 수익성은 subsection predictive method를 사용하여 예측된 수익성을 기반으로 사고 파는 방법을 통해 모델을 운용하여 이에 따른 수익성을 측정 방법으로 사용하였습니다. 수익성의 경우 기존에 사용되는 간단한 방법인 buy and hold 방법론을 적용한 것도 측정 했습니다. 결론적으로는 WSAEs-LSTM 방식이 기존 방법론에 비해 월등한 성능을 갖는걸로 측정 되었습니다.

# Methodology

WSAEs-LSTM는 3단계로 구성 되어 있습니다. 1.Wave Trasnformation, 2.SAE, 3.LSTM 입니다. Wave Transformation을 통해 time series 데이터의 noise를 제거합니다. 그 후 SAE를 통하여 feature를 생성한 뒤, 그 feature들을 이용하여 LSTM을 한 단계 미래의 주가를 예측하도록 트레이닝 합니다.

## Wave Transform

Wave Transformation은 non-stationary한 financial data의 노이즈를 제거하기 위해서 사용 됩니다. non-stationary하다는 말은 데이터의 특정 값의 발생 확률 값이 고정되어 있는 값이 아니고, 시간에 따라서 변할 수 있다는 의미 입니다. 주가 데이터의 경우 특정 가격이 되는 확률이 지속적으로 변하는 값으로서 non-stationary한 데이터라고 할 수 있습니다. wave transformation의 중요한 장점 중 하나는 fourier transform에 비해서 시간에 따라서 주파수 성분을 분석 할 수 있다는 점 입니다. fouriere transform은 특정 구간의 파형이 어떠한 주파수 성분으로 이루어져 있는지를 분석 한다는 점에서 wave transform과 다른 점이라 할 수 있습니다.

여기서는 **Harr** Function을 wave transform의 basis function으로 사용 하였습니다. **Harr** function은 time series data를 cost efficiency하게 분해 할 수 있습니다. Computation complexity는 O(n)입니다. 

추가적으로 financial time series data를 high frequency와 low frequency의 basis function의 합으로 구성하였습니다. 한개의 low frequency basis function과 여러개의 high frequency basis function을 통해서 rough한 time series data를 가공하였습니다. 주가 데이터는 워낙 rough하여 여기서는 wavelet transform을 2번 적용하여 가공을 하였습니다.

wavelet transform을 적용하면 아래와 같은 결과를 얻게 됩니다.
자세한 설명은 이 논문에서 참조한 http://aip.scitation.org/doi/pdf/10.1063/1.4887692 을 보면 됩니다.
![wavelet transform](https://www.nag.co.uk/images/fig_wavelet_jpy-nzd-01.jpg)







