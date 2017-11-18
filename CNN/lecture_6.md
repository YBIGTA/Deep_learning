> Q. 어떻게 Neural Network를 만들까?
>
>  -  *데이터 및 모델 준비(Setup)*
>
>     *-Activation function*
>
>     *-Data Preprocesssing*
>
>     *-Weight Initialization*
>
>     *-Batch Normalization*
>
>  -  Traning dynamics
>
>     -Hyperparameter optimization
>
>  -  Evaluation

실제로 Neural network를 구현할 때 좋은 예측을 위해서는 어떤 것들을 고려해야 하는 지 확인해보자

1. Activation function

   사용할 수 있는 activation function의 종류는 여러가지가 있다.

   ![img](https://i.imgur.com/Va9PTK6.png)

   ​

   > 좋은 activation function의 조건:
   >
   > -functions의 기울기가 평평한 곳이 없어야 한다
   >
   > -activation function의 결과값이 양수,음수 모두

   만약 기울기가 평평한 곳이 있다면(x=10이라고 가정), downstream으로의 gradient가 0이 된다. 이는 결국 gradient가 소멸되게 만들수 있다.

   ![img](https://i.imgur.com/cw2B3x7.png)

   또한 위의 그래프와 같이 activation function의 함숫값이 모두 양수라면, downstream으로 넘어가는 X값은 항상 절댓값을 증가시키는 방향이 된다. 만약 양수였다면 계속 커지고, 음수였다면 계속 작아진다. 따라서 gradient update의 진행 방향이 한정적일 수 밖에 없다. 따라서 파란색 화살표와 같은 w를 얻고 싶을 때는 지그재그로 움직여서 매우 비효율적으로 w를 구하게 된다.![img](https://i.imgur.com/4WaWeYr.png)

2.  Data Preprocessing

   > -0 centering
   >
   > -normalization

   Data를 원점으로 맞추고, 각 차원의 데이터가 동일한 범위 내의 값을 갖도록 정규화 시켜 준다. 이미지의 경우 대부분 픽셀이 0-255사이의 값을 가지고 있기 때문에 normalization 과정은 반드시 필요하지는 않다.

   ![img](https://i.imgur.com/Fz2omJW.jpg)

   실제 전처리 과정에서는 PCA(Principal Component Analysis)와 whitening이 쓰인다. 자세한 것은 아래의 사이트를 참고

   http://aikorea.org/cs231n/neural-networks-2-kr/(cs231n 강의 노트 한국어 번역 )

   http://cs231n.github.io/neural-networks-2/(cs231n 강의 노트)

3. Weight Initialization

   처음 training을 시킬 때는 사용자가 weight을 정해주고, 2번째부터 gradient를 이용해 최적화를 시킨다. 처음 넣어주는 weight에 따라 결과값이 천차만별이 되기 때문에 첫weight을 설정해주는 방법에는 여러가지가 있다. 

   만약 w을 모두 0으로 설정해둔다면, 신경망 내의 모든 뉴런은 동일한 결과를 내게 되고 결과적으로 모두 같은 parameter로 업데이트가 된다. 이렇게 하면 가중치를 주는 의미가 없어진다.

   따라서 고안해낸 방법이 0에 가까운 작은 난수를 weight로 넣는 것이다. 이 때 주의할 점은 만약 신경망이 deep해진다면 downstream으로 가면서 0에 가까운 작은 수들이 곱해져 gradient가 감소할 수도 있다는 점이다.

   반대로 큰 난수를 weight로 넣는다면 각 노드에서 downstream으로 넘어가는 값이 커지게 되어 결국엔 결과값의 격차가 매우 큰 상태가 되어버린다.

   weight를 초기화하는 좋은 방법 중의 하나는 Xavier Initialization이다.  위의 방법들은 입력 데이터수가 커지면 분산도 커진다는 단점이 있는데, 여기서는 가중치 벡터를 입력 데이터 수의 제곱근 값으로 나누어 뉴런 출력의 분산을 1로 정규화한다. 특히 ReLu를 activation function으로 쓴다면 w = np.random.randn(n) * sqrt(2.0/n)로 가중치를 초기화 하는 것이 권장된다. 

    [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv-web3.library.cornell.edu/abs/1502.01852) by He et al (이유는 잘...)![img](https://i.imgur.com/0unx4T9.jpg)

   ​

4. Batch Normalization

   모델에 Traning data 전체를 한꺼번에 학습시키면 부담이 크기 때문에 **작은 mini-batch 단위로 만들어** 학습시키면서 parameter를 업데이트 한다. 전체에 대해 학습 시키는 것이 효과가 최고겠지만, mini-batch로 나누어 학습시키면 **더 자주, 빠르게 parameter를 업데이트** 할 수 있어 충분히 효과적이다.

   이때 mini-batch는 전체를 대표할 수 있도록 만들어주어야하며, Normalization을 해 주어야 한다.

   *자세한 정보는 아래 블로그 참조

   https://m.blog.naver.com/PostView.nhn?blogId=laonple&logNo=220808903260&proxyReferer=https%3A%2F%2Fwww.google.co.kr%2F

![img](https://i.imgur.com/uL8EedM.png)

> Q. 좀 더 효율적으로 학습하는 방법?
>
> :학습과정을 지켜보면서 값들(hyperparameter)을 조정해보자! (Babysitting the learning process)

과정: Preprocess data --> choose architecture

*중요 개념

​	-epoch: 각 자료가 몇 번이나 학습에 사용되었는지

​	(반복횟수는 배치 사이즈 선택에 따라 임의로 바뀔 수 있음)

 

1. Loss 를 확인하자.

   :epoch이 지남에 따라 loss가 어떻게 변화하는 지 확인하자.

   ![img](https://i.imgur.com/7FMdF55.jpg)

2. Train/Validation accuracy

   ![img](https://i.imgur.com/SEeRITV.jpg)

3. Parameter optimization

   :cross validation, learning rate..

   :grid search보다는 random search

   ![img](https://i.imgur.com/ZjenkBT.png)

<Recommendation>

-Activation Function: ReLU

-Data Preprocessing: substract mean

-Weight Initialization: Xavier init

-Use Batch Normalization

-Babysitting the Learning process

-Hyperparameter Optimization

