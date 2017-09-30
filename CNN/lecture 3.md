> Q. 어떻게 최적화된 W를 찾을 것인가? 

##### 				with loss function / optimization

*loss function

: 얼마나 W가 결과를 잘 예측할 수 있을 지 판단하는 기준

: 예측값과 실제 라벨 사이의 차이를 loss function으로 계산한 후, 평균낸 값
$$
L = \dfrac{1}{N}\Sigma{Li(f(xi,W), yi)}
$$
*optimization

: loss function을 최소화해서 최적의 W를 찾기



1) Loss function

1. ##### multiclass SVM

   :  correct score과 incorrect score를 비교하는 것이 핵심.

    정해 놓은 한계(margin)보다  correct score가 incorrect score보다 큰 경우, loss는 0, 만약 아니라면 incorrect score-correct score+1

   ![ img](https://i.imgur.com/qUVlLLg.png): 고양이 사진을 가지고 예측을 한 경우 고양이 카테고리에 해당하는 점수가 2, 개 카테고리가 5.1, 개구리 카테고리가 -1.7이라면 고양이-개, 고양이-개구리를 비교한다. 고양이과 개 사이는 이 되고 고양이-양의 경우 고양이가 훨씬 스코어가 높으므로(차이 > margin) loss는 0이다. 이 스코어들을 다 더한 2.9가 loss가 된다. 

![img](https://i.imgur.com/kb3GEsL.png)

: 만약 correct score가 다른 모든 score에 대해 margin이상 + 차이가 난다면 minimun loss는 0이지만, 만약 그 반대라면 loss는 무한대가 될 수 있다.

Q. 너무 training data에만 맞게 되지 않을까?

[![img](https://i.imgur.com/nFZFZpX.png)](https://i.imgur.com/nFZFZpX.png)

​	-->Regularization으로 모델이 가변적인 test data에도 적용될 수 있게 만든다. (Make model simple!)

​	type) L1, L2 regularization



2. ##### softmax classifier (Multinomial Logistic Regression)

   : 각 score를 확률로 나타내고, 이를 -log취해 lost function을 구한다.

![img](https://i.imgur.com/IFvv06u.png)

​	:probability가 1에 가까운 경우, loss function은 0에 가깝고 probability가 작으면 작을수록 loss fuction은 무한대가 된다.



##### *multinomial SVM	vs	softmax classifier

![img](https://i.imgur.com/7lJ1Xad.png)

:차이점?

softmax는 끊임없이 correct class의 확률을 1로 수렴시키려고 한다.

(SVM은 correct와 incorrect 사이에 어느 정도 차이가 많이 난다면 그만..!)



> Q. 어떻게 최적의 W를 찾을 것인가?

2)Optimization

: loss function이 최소화 되는 점 찾기..!

1. ##### Random search

   : 랜덤으로...! 시간도 오래 걸리고..운에 맡긴다:(

2. ##### Gradient descent

   :  gradient를 점점 줄여나가면서 minimum W 값을 구한다.

   ![img](https://i.imgur.com/fjS67ut.png)

   Q. step size( = learning rate)?

   ​	: hyperparameter이다. gradient를 얼마만큼씩 줄여나갈 건 지 결정하는 요소.



하지만, 학습데이터가 매우 많다면(N이 매우 크다면) W를 업데이트 할 때 마다 시간이 엄청 오래걸린다.

==> Stochastic Gradient Descent(SGD) 사용

:핵심은 학습 데이터의 일부인 mini batch만을 이용해 gradient를 구하고 parameter(W)를 업데이트 하는 것.

:미니 배치를 이용해도 전체를 구하는 것과 비슷할 것이라고 가정한다. 미니 배치를 이용해서 구하면 W를 더 자주자주 업데이트 할 수 있어 더 빠르게 W의 최솟값을 구할 수 있다.