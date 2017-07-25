YBIGTA 10기 노혜미

# 단어, 문장을 vector로 표현하기

## 왜 굳이...?

'그냥 단어 자체를 넣으면 되는 거 아닌가?'라고 생각할 수 있다.

하지만 해당 단어를 처리하는 것은 '사람'이 아니라 '컴퓨터'이다.

컴퓨터가 단어를 인식하기 위해서는 단어를 수치화하는 방법이 필요하다!

특히 딥러닝은 '학습'을 통해 '예측'을 하기 때문에 더더욱 필요하다.

## one-hot vector

'모두의 딥러닝'의 sung kim 교수님이 말하시길 1인 부분이 핫(?)하기 때문이라고 말씀하셨다.

구구절절 설명하는 것보다 예시를 들어서 설명하는 게 훨씬 빠를 것이다.

['와빅', '최고얌']이런 단어 리스트가 있다고 할 때, '와빅'은 [1, 0], '최고얌'은 [0, 1]의 vector로 표현된다.

보통 '가나다'순으로 정렬되는 듯 싶다.

하지만 이런 one-hot vector는 문제가 있다...!

단어간의 유사도라든지 단어 간의 관련성을 알아낼 수 없다.

['고양이', '멍멍이', '아이스크림']를 one-hot vector로 나타냈을 때 다음과 같다.

고양이 = [1, 0, 0] / 멍멍이 = [0, 1, 0] / 아이스크림 = [0, 0, 1]

고양이와 멍멍이는 살아있는 생물이라는 공통점이 있다. 그래서 문장의 주어로도 쓰일 수 있을 것이다. 아이스크림은 그렇지 못하다.

하지만 고양이&멍멍이 고양이&아이스크림 간의 차이는 없다...!

컴퓨터에게는 모두 이러한 특징, 즉 문맥이 싸그리 무시된다. 

(간단하게 해당 nlp(nature language processing) 모델을 구현해 보고 싶을 때는 유용할 것이다.)

그래서 여러 과정을 거쳐 아래와 같이 문맥이 고려되는 neural embeding을 이용한 word2vec이 등장한다.

## word2vec model

word2vec model에는 크게 2가지가 있다.

**CBOW**와 **Skip-gram**이다.

딱 단어들만 봤을 때는 느낌이 오지 않는다. (이 분야는 줄임말도 많고 처음보는 단어도 엄청 많다...)

밑의 이론을 통해 자세히 알아보자.

## CBOW(Continuous Bag-of-Words)

### one-word context

하나의 context word가 주어지면 하나의 targer word를 예측하는 모델이다.

'이런 맥락에서 뭐가 쓰인거지?'라는 궁금증을 해결해주는 모델이라고 생각하면 될 것 같다.

![one-word context](http://i.imgur.com/NG98Oht.jpg)

layer에 대한 설명은 생략한다...그냥 '들어가는 레이어고 중간에 있는 숨겨진 레이어고 나오는 레이어구나~'정도로 알면 될 것 같다.

input으로 들어가는 것은 단어를 수치화한 vector이다. 이 vector는 위에서 언급했던 one-hot vector이다.

V개의 element(V개의 단어)중 해당 context word만 1이고 나머지 word는 0이 된다.

'그러면 단어 1개 밖에 신경 못쓰는 거 아닌가?'라는 의문이 들 수 있다. 하지만 그렇지 않다. 저런 V행 vector가 그때그때 갱신된다. 아웃풋도 마찬가지이다.

예를 들어 context word가 'bark'이고 target word는 'dog'이다. 

학습 과정 중 모델은 'bark'를 입력으로 받았다. 그래서 'dog'가 나올 확률을 0.7, 'cat'이 나올 확률을 0.2라고 예측했다. 이 확률을 y_i라고 한다. 

그리고 t_i라는 것이 등장하는데, output word가 실제 target word이면 1을, 그렇지 않다면 0을 값으로 가진다.

y_i에서 t_i를 뺀 것을 에러로 정의한다. 만약 y_i가 더 크다면 이는 실제 단어가 아닌데 모델이 잘못 예측한 것이고(overestimating), t_i가 더 크다면 실제 단어가 맞는데 모델이 잘못 예측한 것이다. 뭔 소리인지 감이 안 온다면 예시로 다시 돌아가보자! 

'dog'는 실제 target word가 맞으므로 'dog'의 확률 0.7 과 1을 빼준다. 이때 t_i값이 더 크게 된다. 모델이 좀 더 dog라고 예측해 줄 필요가 있다. 그래서 'dog'의 vector를 'bark' vector에 가깝해 만들어 준다. 'cat' vector의 경우는 반대로 'bark' vector와 멀게 만들어 준다. y_i와 t_i의 차에 따라 가깝고 멀고를 조절한다.

### multi-word context

one-word context와 달리 여러 개의 context word에서 하나의 target word를 예측하는 모델이다.

계산하는 방법 같은 것만 좀 다를 뿐, 근본적으로 같은 것 같다...

어떤지 그림으로 느낌만 갖고 가자.

![multi-word context](http://i.imgur.com/NCdtgY4.jpg)

# skip-gram

위에 나온 CBOW 모델과 정반대라고 생각하면 된다.

target word가 input이 되고 context words가 output이 된다.

'이 단어가 뭐랑 쓰였대?'라고 궁금해 한다면 skip-gram 모델을 쓰면 된다.

c-bow보다 비교적 많은 데이터를 필요로 한다고 한다.

그림으로 어떤 모델인지 살펴보자.

![skip-gram](http://i.imgur.com/UFRHTuB.jpg)

skip-gram도 사실 근본적으로 CBOW의 one-word-context와 크게 다르지 않은 것 같다.

에러를 구하는 것도 상당히 유사하고 "The derivation of parameter update equations is not so diferent from the one-word-
context model."라고 해당 논문에서 나오니까 말이다.

좀 더 구체적으로 알고 싶다면 밑의 skip-gram을 tensorflow로 구현하기 전 설명을 참고하자!

one-word-context 모델만 잘 이해해도 반 이상은 이해했다고 할 수 있을 것 같다.

## word2vec 직관적 이해

https://ronxin.github.io/wevi/

설명: https://www.youtube.com/watch?v=D-ekE-Wlcds

*Neurons*

- one input layer, one output layer, one hidden layer
- red: excited
- blue: inhibited

*Weight Matrices*

- input vector: weights between the input layer and the hidden layer
- output vector: weights between the output layer and the hidden layer
- more red: more positive
- more blue: more negative

*Vectors*

- blue dots: input vectors
- orange dots: output vectors

## gensim

다 됐고 그냥 단어를 벡터로 바꾸고 싶다면 gensim이라는 라이브러리를 쓰는 것을 추천한다.

신기하게 메소드 몇 개 돌려주면 결과가 나온다...!

gensim에 대한 튜토리얼을 보고 싶다면

https://rare-technologies.com/word2vec-tutorial/

이 사이트를 참고하길 바란다.

만약 위 사이트와 달리, gensim으로 한국어 텍스트를 처리하고 싶다면 konlpy라는 패키지를 깔아서 형태소 태깅을 해야한다.

그래야 좀 더 자세히 단어를 분류할 수 있기 때문이다. (사람들이 많이 쓰는 건 Twitter 같다.)

구현된 것의 결과가 어떤지 보고 싶다면

http://w.elnn.kr/search/

방문해보길 바란다.

'치킨+콜라 = 피자'의 결과가 나온다.

치킨을 먹을 때는 피자도 먹길 바란다.

### reference

- https://www.lucypark.kr/slides/2015-pyconkr/#11
- Xin Rong,word2vec Parameter Learning Explained(2014)
  https://arxiv.org/pdf/1411.2738v4.pdf
- https://radimrehurek.com/gensim/models/word2vec.html
- https://rare-technologies.com/word2vec-tutorial/
- http://w.elnn.kr/search/
- https://ronxin.github.io/wevi/
- https://www.youtube.com/watch?v=D-ekE-Wlcds