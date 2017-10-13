## Backpropagation

> Q. Gradient를 어떻게 조절할 것인가?
>
> ​	A. Backpropagation을 이용해서!

지난 시간, Loss function을 최소화 시키기 위한 Weight를 조정하기 위해 Gradient descent를 이용한다는 사실을 배웠다. 

Gradient는 loss를 줄이기 위해 weight 벡터를 조정할 때 어떤 방향으로 움직여야할 지를 알려주는 지표이다.  Gradient가 0이 되는 global minimum을 찾아 loss function을 최소로 만드는 Weight을 찾는 것이 목적이다.

![img](https://i.imgur.com/1OJghYx.png)



매우 복잡한 수식을 다룬다면 gradient 계산은 매우 어려워지는데, Backpropagation은 **복잡한 수식을 쪼갠 다음 chain rule을 이용해 엮어서** 간단하게 gradient를 계산하는 방법이다.

​	*chain rule
$$
\dfrac{\partial{f}}{\partial{x}} = \sum_{i}\dfrac{\partial{f}}{\partial{p_i}} *\dfrac{\partial{p_i}}{\partial{x}}
$$


아래 그림에서 볼 수 있듯이 매우 복잡한 수식을 간단한 연산으로 나누고 각 연산을 노드로 정의하고 나면 복잡한 수식은 간단한 노드들이 엮인 모양이 된다.  

각 노드가 최종 결과값에 미치는 영향을 알아낸 후 (각 노드의 gradient) 이를 엮으면(chain rule) 변수와 weight에 따른 최종값의 변화를 알 수 있다. 각 노드의 *Gradient*는 **그 노드가 최종값에 영향을 미치는 민감도**를 나타낸다.![img](https://i.imgur.com/pWPpDiI.png)

각각의 노드에서 이루어지는 연산은 다음과 같다. 각 노드의 입력값이 결과값에 얼마나 영향을 주는 지 local gradient를 통해 확인한다.

![img](https://i.imgur.com/pEBaEGc.png)



예시)

![img](https://i.imgur.com/0e2V3Rd.png)

Q. 숫자가 아니라 행렬,벡터로 주어진다면?![img](https://i.imgur.com/kiKKjwK.png)



Q. Jacobian matrix?



## Neural Networks

**여러 층의 layer**을 사용해 , 중간중간에 **Non linear function을 이용**해서 필터링을 해 주는 형태이다.

Non linear function은 여기서 나온 것처럼 0이하의 값을 다 0으로 바꾸어 버리는 function일 수도 있고, ReLu, ELU, sigmoid등 다양한 함수를 사용할 수 있다.

Q. 여러 층의 layer을 사용하는 이유는?

input을 계층적으로 분석가능하기 때문이다. 여기서 계층적이라는 말의 의미는 다음과 같다. 예를 들면 페르시안 고양이, 스핑크스 고양이, 삼색 고양이를 각각 분류하고나면 이들을 고양이라는 카테고리로 묶을 수 있고, 고양이는 또한 동물이라는 큰 카테고리 안에 있다. 따라서 삼색 고양이 사진을 넣는다면 '삼색 고양이는 고양이의 일종이고 고양이는 동물의 한 종류이다'라는 흐름을 거쳐서 '삼색 고양이는 동물이다'라는 결론을 낼 수 있다는 것을 의미한다.

![img](https://i.imgur.com/NjrbmG9.jpg)

가령 input으로 동물사진들이 주어지고, 동물들을 분류하는 것이 목적이라고 생각해보자. 

1번째 layer에서는 1차적으로 들어온 사진들을 그대로 분석하기 때문에 말의 오른쪽 얼굴, 왼쪽 얼굴을 각각 다른 클래스로 분류해 트레이닝 한다. 따라서 만약 정면을 보고 있는 말 사진을 넣는다면 말의 오른쪽 얼굴, 왼쪽 얼굴에 해당하는 클래스의 스코어가 50,50으로 비슷하게 나올 것이라고 생각해볼 수 있다. 

여기서 2번째 layer을 넣는다면 1차적으로 계산된 스코어를 다시 엮어서 2차 weight를 구할 수 있다. 말의 오른쪽, 왼쪽 얼굴에 해당하는 score들을 엮어 2차 연산 후에는 '말'이라는 카테고리가 가장 점수가 높도록 할 수 있다는 점이다.



Q. Non linear function이 왜 중요할까?

어떤 것이 유의미하고 아닌지 한번 필터링 해주는 역할이 있다.  무의미한 값들은 score을 0으로 만들어버리거나 아주 작게 만들어버리고, 유의미한 값들은 score를 높게 만들면 유의미한 값들만 2차 layer에서 계산할 수 있다는 장점이 있다.

![img](https://i.imgur.com/DuLj3Sm.png)







