YBIGTA 10기 노혜미

# Tensorflow Optimizer 안내서

이제 막 텐서플로우를 사용하기 시작했다면, 골치 아픈 게 매우 많을 것이다.

그중에 optimizer 분명 있을 것이다.

너무 많아서 어떤 걸 써야할지 고민이 된다면...

이 글을 찬찬히 보길 바란다 : )



*참고

코드로 밑의 optimizer들을 쓰고 싶다면

> tf.train.(해당)Optimizer

라고 쓰면 된다! 

(물론 learing rate같은 것을 설정해줘야 한다.) 



## Momentum

- SGD의 약점 보완해준다.
- 좀 더 빠르고 왔다갔다 하는 것을 완화해준다. 무슨 말인지 와닿지 않으니 밑의 사진들을 보자!

> momentum이 없는 SGD



![without momentum](http://i.imgur.com/TfPIlPX.gif)

> momentum이 있는 SGD

![with momentum](http://i.imgur.com/UgeumXK.gif)


## Adagrad

- gradient 기반 optimization이다.
- sparese data 즉, 드문드문한 데이터를 다루는데 적합하다고 한다.
- (갓)구글에서 큰 규모의 신경망(neural net)을 학습시키는데 이용했다고 한다.
- 그리고 Glove논문의 저자 중 한 명인 Pennington이 GloVe word embeddings를 학습시키는 데 사용했다고 한다. 빈도수가 떨어지는 단어는 빈도수가 많은 단어보다 훨씬 더 업데이트되야 하기 때문이라고 한다.
- 결국 빈약한(?)데이터에 안성맞춤인 optimizer라고 할 수 있다.

## Adadelta

- Adagrad의 확장판이다. 
- 구글과 GloVe 개발자가 써서 아무 문제가 없었을 것 같았는데, Adagrad도 문제가 있었다. 수렴될 정도로 학습율(learning rate)을 줄였던 것이다. 
- Adadelta는 욕심쟁이처럼 제곱된 모든 gradient를 누적시키는 대신에, 어떤 고정된 사이즈 w까지만 과거* gradient를 누적한다.
- 모든 과거의 제곱된 gradient를 exponentially decaying average*시킴으로써 gradient의 합이 반복적으로 정의된다. 

*아직 모델을 학습시켜보지 않은 사람은 과거라는게 무슨 뜻인지 감이 안 올 수 있다. 모델은 학습할 때, 자신이 갖고 있는 변수들(w, b)을 계속계속 업데이트한다. 즉 순차적으로 변수들의 값이 바뀌기 때문에, '과거(past)'라는 말을 쓰는 것이다.

*decaying average는 모든 데이터를 동등하게 생각하지 않는 방법이다. 과거의 데이터도 고려하긴 하지만, 현재의 데이터를 더 중요시한다. 

## AdagradDA

- Adagrad가 sparse data에 적합했던 것처럼, AdagradDA 역시 학습된 모델에서 large sparsity가 있을 때 필요하다. (데이터에 큰 구멍(?)이 뚫려있을 때 적합하다고 생각하면 될 것 같다.)
- 다만 AdagradDA는 선형모델(linear models)에 대한 sparsity에만 적합하다.
- 또 주의할 점은 학습시킬 때, gradient accumulator의 초기화를 잘해야 된다는 것이다.

## RMSprop

- Aadelta와 비슷한 optimizer이다.


- 유일한 차이는 cache라는 요소를 어떻게 계산하냐이다. 


- 1* - (제곱된 gradient의 과거 합)을 현재 gradient로 본다.*


- 그래서 축적된 gradient가 평균 정도에서만 움직이기 때문에 발산할 일이 없다.

* 여기서 1은 전체 데이터를 의미한다. 
* Adadelta는 cache를 어떻게 계산하는지 찾아봤지만 나오지 않았다...아마 RMSprop과 달리, 현재 구한 gradient를 그대로 넣는 방식이지 않을까 추측한다.

## Adam

- 실제로 가장 많이 쓰이는 optimizer이다. 

(모두의 딥러닝에서도 교수님이 이걸 쓰라고 하신다.)

- RMSprop의 수정버전이다. momentum이 있는 RMSprop이라고 생각하면 된다.
- Adadelta와 RMSprop처럼 과거의 제곱된 gradient의 exponentially decaying average(v_t)만 저장하는 것이 아니라 제곱되지 않은 것(m_t) 또한 저장한다.
- v_t와 m_t는 처음의 몇 번째 step에서 0으로 편향되기 때문에, 처음부터 아예 0으로 초기화시켜준다. 그래야 최종적으로 0으로 편향되는 것을 막을 수 있기 때문이다.

## Optimizer Test시각화

*다루지 않은 optimizer는 생략하겠다.

x,y축은 weight이고 z축은 loss이다. 

여기서 테스트 하고자 하는 것은 optimizer들이 특정 구간(local minumum)에 빠지지 않고 loss를 가장 최소로 만드는 값을 찾아내냐이다.

즉, global minimum을 찾아내는 가를 테스트하는 것이다.

좀 더 쉽게 말하자면, optimizer가 함정에 빠지지 않고 진짜를 찾아내는가 시험해보는 것이다.

> at long valley

![long valley](http://i.imgur.com/RM5QeGt.gif)

1. Adadelta
2. momentum
3. RMSprop
4. Adagrad
5. SGD



> at Beale's Function

![beals's function](http://i.imgur.com/XmKqJJz.gif)

(별과 접촉이 기준)

1. Adadelta
2. Momentum
3. RMSprop
4. Adagrad
5. SGD



> at saddle point

![saddle point](http://i.imgur.com/Jo3nbgn.gif)

1. RMSprop
2. Adagrad
3. Momentum
4. Adadelta
5. SGD



다른 건 몰라도 SGD는 안 쓰면 될 것 같다. 

아쉬운건 제일 널리 쓰이는 Adam이 없다...

## Conclusion

- 왠만하면 Adam을 쓰고 성능이 그렇게 좋지 않다면 Adadelta로도 한 번     써보는 것을 추천한다.
- 자신이 sparse한 데이터를 갖고 있다면 Ada계열 혹은 RMSprop를 써보자.
- 사실 많은 글 들에 이럴 때는 이런 걸 써라! 이런 식으로 나오지 않았다.      그래서 안내서지만, 정확한 가이드 라인을 제시하지 못 한 것 같아 아쉽다...

## Rerference

http://ruder.io/optimizing-gradient-descent/index.html

http://wiseodd.github.io/techblog/2016/06/22/nn-optimization/

http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html

http://blog.csdn.net/lfglfglfglfg/article/details/53583785

http://donlehmanjr.com/Science/03%20Decay%20Ave/032.htm
