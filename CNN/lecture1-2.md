> 컴퓨터가 이미지를 인식하고, 분류하게 하고 싶다.

#### Q. 컴퓨터가 이미지를 인식하는 방식은?

이미지를 픽셀 단위로 나누고 각 픽셀의 RGB값을 숫자로 나타낸 것을 인식한다.

![img](https://i.imgur.com/NgL738P.png)

 

##### Challenges

1.    한 대상에 대한 이미지임에도 빛에 따라 다르게 보인다


2.    한 대상이 여러 형태를 보인다

      eg) 앉은 고양이, 누워 있는 고양이

3.    대상의일부만 보인다

4.    배경이랑 구분이 어렵다. 

5.    같은 카테고리에 속하더라도 개별적인 차이가 있다

      eg) 삼색 고양이, 스핑크스 고양이는 고양이로 분류 가능하지만 특징이 매우 다르다 

 --> 이와 같은 문제점을 극복해야 이미지 분류를 잘 할 수 있다



##### Q. 이미지를 분류하기 위한 방법?

####  : Data-Driven approach

--> 이미지를 분류하기 위한 모든 기준들을 정해두고 분류하기보다는,  많은 양의 데이터를 가지고 학습시켜서 새로운 이미지가 오더라도 잘 분류하도록 만들자.

Q. hyperparameter란?

:컴퓨터가 아니라 쓰는 사람이 지정해야 할 세팅들



Q. 학습시킬 때 내가 얼마나 잘 학습시켰는 지 어떻게 알지?

: train, validation, test set으로 데이터를 나눈다. training set으로 학습시키고 validation set으로 알고리즘이 잘 작동하는 지 파악한 후 valdiation set에 쓰인 hyperparameter을 가지고 최종적으로 test set으로 결과를 본다. ![img](https://i.imgur.com/IETYIA5.png)

*** cross validation 

:어떤 hyperparameter가 좋은 결과를 낼 지 모르기 때문에 다양한 validation set을 이용해서 hyperparameter를 선택한다. 이 때 validatoin set은 training set을 여러개로 쪼갠 것 중 하나이다. 

-->deep learning에서는 시간이 너무 오래 걸려서 쓰지는 않는다

![img](https://i.imgur.com/vgOeygU.png)





1. ##### k-Nearest Neighbor

   : 라벨을 알고 있는 각 데이터를 벡터화 시켜 좌표 평면에 점을 찍는다. (eg. 동물사진을 분류하는 경우라면 어떤 동물인지 알고 있는 데이터) 이후 새로운 이미지가 들어왔을 때,  이 점을 중심으로 k개의 가장 가까운 점을 찾은 후 이 점들이 어떤 분류에 속하는 지 본다. 이 때 새로 들어온 이미지는 가장 높은 표를 받은 분류에 속한다고 생각한다.![img](https://i.imgur.com/mkW8Jvd.png)

​	Q. 이미지가 비슷한 지 어떻게 비교하지?

​		distance를 이용

​		1) L1 distance

​		2) L2 distance

​			==>완벽히 이해 못했지만..이후 강의에서 다시 자세하게 설명함!



​	**여기서의 hyper parameter

​		: k / L1 or L2 distance / validation or crossvalidation



문제는, KNN이 deep learning에서 잘 쓰이지 않는다는 것이다.

이미지는 너무 데이터의 차원이 높다. (수많은 픽셀들과 각각에 대한 RGB값..!)

또한 이미지간의 distance가 직관적이지 않다. 예를 들어 한 이미지를 미세하기 수정한 경우(eg. 톤을 미세하게 조정했다거나, 이미지의 일부를 가린 경우) L2 distance가 변하지 않는다. 분명히 이미지를 변화시켰는데, 컴퓨터는 이들을 같은 여전히 distance를 가지는 것으로 인식한다. 변한 것도 잡아내야 좋은 이미지 분류다..!



KNN대신 쓸 수 있는 알고리즘?

2. ##### Linear Classification

   : 핵심은 parametric approach

   : KNN은 training data에 대한 정보를 test할 때에도 그대로 가지고 있어야 한다. 하지만 linear classification 에서는 training data에 대한 정보는 W라는 parameter로서 압축해서 사용하고 test할 때는 W만 가져와서 계산하기 때문에 경졔적이다.

   : X라는 데이터를 모델에 집어넣고 학습을 시키면 원하는 결과값을 얻을 수 있도록 W(weight)가 조정된다. 이 때 결과값(score)를 계산하는 함수가 f(w,x) = Wx +b로 일차 함수의 형태를 가진다. b(bias)는 특정 카테고리의 data independent preference를 고려하는 요소이다.

   예를 들면, 고양이 사진을 넣었을 때 고양이,개,양 세 종류의 카테고리로 분류할 수 있는 모델이 있다. 이 때 고양이에 해당하는 점수가 가장 높다면 고양이로 판별한다. 고양이 사진을 컴퓨터가 이해할 수 있는 숫자 형태로 바꾼것이 x이다. W는 '고양이를 고양이로 판별할 수 있게 조정된 하나의 매트릭스'로서 강아지가 들어오면 강아지로 판별할 수 있게 한다. 만약 데이터에 고양이 사진보다 강아지 사진이 많은 경우, 데이터 자체의 값에 관계없이 강아지로 판별할 가능성이 높게 만들어야 한다. 이 때 사용하는 것이 b이다.

   : x --> wx + b --> score

   ##### ![img](https://i.imgur.com/jllkY2j.png)

​	linear classifcation은 쉽지만 분류를 잘 할 수 있는 알고리즘이다.

​	하지만, linear classification은 데이터의 분포에 따라 분류를 못하기도 한다.

![img](https://i.imgur.com/NMputwR.png)

​	왼쪽과 같이 데이터가 분포되어 있을 때는 선 하나로 두 그룹을 잘 나눌 수 있지만 나머지는 그럴 수 없다.



