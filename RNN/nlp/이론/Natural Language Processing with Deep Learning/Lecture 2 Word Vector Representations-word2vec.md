<h1>Lecture 2 | Word Vector Representations: word2vec</h1>
강의 출처:<br>
https://www.youtube.com/watch?v=ERibwqs9p38&index=2&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6

![1](http://i.imgur.com/ZIEho7v.png)
- 단어를 vector로 나타내는 방법에는 one-hot이라는 방법이 있다. 이는 단어 개수 만큼의 차원을 가진 vector에서 해당 단어의 index에 해당하는 부분을 1로 하고 그 외에는 0으로 표현한다.
- word set이 별로 없을 때는 괜찮지만, word set이 매우 많을 때, one-hot으로 단어를 표현하게 될 경우, word vector의 차원은 엄~청 커진다.

![2](http://i.imgur.com/X2Y0GWY.png)
- 게다가 단어 간의 유사도를 측정하기 어려워진다.
- 위의 예시에도 나오듯이 motel과 hotel은 비슷한 단어이지만, 서로 곱해도 결과는 0이다.

![3](http://i.imgur.com/Bjtrh6C.png)
- 어떤 단어 즉, 단어의 의미를 알기 위해서는 그 단어의 주변을 보면 안다. 단어가 어떤 맥락에서 쓰였는 지를 살펴보면 되는 것이다.
- 예를 들면 'banking'이란 단어는 problem, crisis, regulation과 같은 단어와 쓰인다.

![4](http://i.imgur.com/SAf6iCN.png)
- 단어를 나타내기 위해 one-hot을 쓰는 대신에 dense vector를 쓸 수도 있다.(사실 이게 훨씬 많이 쓰인다.)
- 해당 context에 등장하는 다른 단어를 예측하기에 좋다.

![5](http://i.imgur.com/O9LHKN0.png)
- center word $w_t$와 context words를 예측하는 게 모델의 목표이다.
- 이때 loss function을 갖는데, -t는 target word주변에 있는 단어, 즉  context words를 가리킨다. $p(w_{-t}|w_t)$는 target word의 주변에 context word가 올 확률이므로 이 확률이 높아질 수록 loss는 작아진다.
- loss를 줄이기 위해 단어의 vector representation을 계속해서 조정하는 게 필요하다.

![6](http://i.imgur.com/aLt8uvQ.png)
- 두 가지 알고리즘과 두 가지 traing methods가 있다.

![7](http://i.imgur.com/8z4Wd7I.png)
- skip-gram model의 아이디어는 각각의 estimation step에서, 단어 한개를 center word로 하고 context words를 예측한다. 이때, window size라는 개념이 등장하는데 그냥 얼마 만큼의 맥락을 고려할 지를 결정하는 것이다.
- model은 주어진 target word의 context word가 나타날 확률 분포를 결정한다.
- 그리고 vector representation을 선택한다. 그래서 확률 분포를 최대로 만들 수 있게 된다.
- 여기서 중요한 점은 단어의 왼쪽, 오른쪽 이런식으로 여러가지의 확률 분포를 갖는 게 아니라, context word에 대한 오직 하나의 확률 분포를 갖는다는 것이다.
- context word는 output이 된다.

![8](http://i.imgur.com/bgrUODD.png)
- objective function*은 어떤 center word가 있을때 해당 center word의 context word의 확률을 최대로 만드는 것이다.
- 여기서 J prime은 아주 긴 sequence of words가 주어졌을 때, t=1부터 t=T(마지막)까지 각각의 단어의 위치를 이동한다. 각각의 위치에서 window를 가지는데, 이 window size는 2m이다. target word의 이전 m번째까지의 단어와 이후 m번째까지의 단어를 고려하겠다는 뜻이다. 그리고 target word에 대한 확률 분포를 가진다.
- model의 유일한 parameter($\theta$)는 단어의 vector representation이다.*
- 각각의 target word에서 context words가 나올 확률에 log를 취하고 이를 window size를 고려해 모두 더해준다. 그리고 이 과정을 T번(모든 단어) 반복해준다. 1/T를 하는 이유는 모든 확률의 평균 값을 내기 위해서이다. 이때 -를 붙여줌으로써 Negative log likelihood를 이용한다. 그렇다면 J($\theta$)가 의미하는 것은 해당 target word 다음에 context words가 나오지 않을 가능성이 된다. 이 가능성을 줄여주는 것이다.
- log likelihood대신 Negative log likelihood를 사용하는 이유는 보통 optimizer는 해당 값을 줄여주기 위해 사용하기 때문이라고 한다.
*objective function=loss function=cost function이라고 한다.
*어떤 학생이 window size도 parameter아니냐라고 질문하는데 교수님이 사실은 자기가 cheat를 썼다며 window size도 사실 hyper parameter라고 한다. 하지만 일단 constant라고 알아두라고 한다.

![9](http://i.imgur.com/dMEASYN.png)
- 여기서 c는 center word의 index이고 o는 context words의 index이다. 그래서 $u_o$는 context word vectors이고 $v_c$는 center word vectors이다.
- 두 word vector를 dot product시킨다. 똑같은 위치에 있는 vector의 element들을 곱해준다 다음 모든 곱의 합을 내는 것이다. 이때 두 vector가 비슷할 수록 합의 값이 커진다. doc product의 결과를 softmax form에 넣는다.

![10](http://i.imgur.com/QLEGjse.png)
- softmax function은 number를 확률 분포로 바꿔주는 함수이다. dot products를 계산하면 그냥 number이다. 그래서 바로 확률 분포로 쓸 수 없기 때문에, 지수화한다. 지수화할 경우, 음의 값이 나올 일이 없기 때문이다.
- 그리고 지수화한 것을 sum으로 나눠주면 정규화를 할 수 있다.
- 이러한 과정을 거쳐서 확률 분포를 반환하게 된다.

![11](http://i.imgur.com/01chqTU.png)
*교수님이 한획 한획 그림판으로 그린 것 같다.
- V는 단어의 개수이고 d는 단어를 vector로 나타낼 때의 차원이다.  
- center word를 one hot vector로 만든다. 그리고 center words의 matrix representation에서 one hot vector의 1에 해당하는 index를 갖는 column을 뽑아낸다. 즉, 해당 center word의 vector representation을 뽑아내는 것이다.
- center word의 vector representation을 context words의 representation을 담고 있는 matrix와 곱해준다.(세 방향으로 뻗어 가는 거)
- center word와 context word를 dot product를 해준다.
- softmax에 넣어주고 확률 분포로 만든어준다. 만약 모델에서 해당 context word일 확률을 0.1로 준다면 1-0.1을 한 값이 loss가 될 것이다.

![12](http://i.imgur.com/hSpFwR6.png)
- 모델에서의 모든 parameter의 set을 하나의 긴 vector인$\theta$로 정의한다.
- skip-gram model에서는 d차원의 dimension을 가지는 v개의 word vectors의 matrix일 것이다.
- 이때 matrix의 길이가 $2dV$인 이유는 하나의 word vector가 center word로 쓰일 수도 context word로 쓰일 수도 있기 때문이다.

objective function의 값을 최소화하기 위해 gradient를 이용한다. 이때 partial derivatives를 하는데 다음과 같은 과정을 거친다.
$$ {\partial\over\partial v_c} log{exp(u^T_ov_c)\over\sum_{w=1}^Vexp(u^T_wv_c)} $$
$$ = {\partial\over\partial v_c}(log{exp(u^T_ov_c)}-log\sum_{w=1}^Vexp(u^T_wv_c))$$
$$\cdots$$
$$= u_o - \sum_{x=1}^Vp(x|c)u_x$$
- 이때 $u_o$는 실제 output context word vector이고 $\sum_{x=1}^Vp(x|c)$는 context에 나타날 수 있는 모든 가능한 단어의 확률들의 합이다. 이때 $u_x$는 expectation vector이다. 발생 가능성으로 weighted되는 모든 가능한 context vectors의 평균이다.
- 위의 두 값이 최대한 가까워지도록 model에서 parameter를 바꾸는 게 우리의 목표이다.

![13](http://i.imgur.com/HMPPBPS.png)
- 해당 window에서 사용되는 모든 parameter 즉, 모든 word representation의 updates를 계산한다.

![14](http://i.imgur.com/pPw1qJt.png)

![15](http://i.imgur.com/vP9ltzO.png)
- gradient descent의 아이디어는 parameter인 $\theta$에서 objective function을 $\theta$로 미분한 값에 learnig rate를 곱한 값을 빼주는 것이다. 이렇게 함으로써, 새로운 $\theta$값이 나오는 것이다.

![16](http://i.imgur.com/JRP33vC.png)
- 이처럼 가장 작은 기울기를 가지는 곳으로 점차 이동하는 것이다.

![17](http://i.imgur.com/mruYo9x.png)
- 하지만 토큰(단어)가 매우 많을 경우 이를 단어 하나하나에 적용한다면, 엄~청 오래 걸릴 것이다.
- 그래서 나온 방법이 SGD이다. SGD는 단어 하나하를 고려하는 것이 아니라 window 하나 씩을 고려하는 방법이다. 전체 단어에 대한 gradient를 계산한 다음 updates를 하는 게 아니라, 각각의 window별로 gradient를 계산한 다음, 각각을 updates하는 것이다.
