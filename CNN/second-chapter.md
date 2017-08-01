# Chapter 2. 불확실성의 언어

## Bayesian modeling

### Likelihood vs Probability

< 특정 x >
불연속 사건 ( 주사위... ) : Likelihood == Probability
연속 사건 ( 신호...? ) : pdf의 y값 == Likelihood || Probability == 0

### Likelihood distribution in Model

$$p(y=d|x,w)=\frac{exp(f^w_d(x))}{\sum_{d'} exp(f^w_{d'}(x))}$$ : Classification tasks

$$p(y=d|x,w)=N(y;f^w(x),t^{-1}I)$$ : Regression....?

#### Predict output for an new input point x* 
In Bayesian modeling, inference == integral
$$ p(y^*|x^*,X,Y) = \int p(y^*|x^*,w)p(w|X,Y)dw $$

$$p(Y|X) = \int p(Y|X,w)p(w)dw $$
여기서 계산되지 못할 Inference는 어떻게 처리를 해야하는가....?

## Variational Inference
Function : 입력이 주어지면 이와 매핑된 값을 출력하는 함수 ( ex - $$f(x)=x^2+3$$ )
Functional : 함수들의 집합을 정의역으로 갖는 함수 ( ex - $$J[y] = \int y(x) dx $$ )
-> ( Functional은 결국 loss 함수...? )
Functional의 미분은 y가 다양하기 때문에 '정확히' 계산하는 것은 불가능하다. 이를 Variational Inference를 통해 해결한다 -> 근사화

## Reference
- http://rstudio-pubs-static.s3.amazonaws.com/204928_c2d6c62565b74a4987e935f756badfba.html [가능도(likelihood) vs 확률]
- http://norman3.github.io/prml/docs/chapter10/0