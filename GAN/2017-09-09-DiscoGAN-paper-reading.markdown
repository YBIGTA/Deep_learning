<span style="color:#7C7877; font-family: 'Apple SD Gothic Neo'; font-weight:200">

# DiscoGAN 논문 이해하기

이번 시간에는 GAN의 응용 논문 중 우리나라 SK T-Brain에서 발표되서 인정받고 있는 Learning to Discover Cross-Domain Relations with Generative Adversarial Networks(이하 Disco-GAN)을 소개하겠습니다. 연세대학교 빅데이터 학회 YBIGTA GAN팀에서는 다음과 같은 순서를 GAN을 공부하고 있으며, 클릭하면 내용을 확인하실 수 있습니다.

**Study 순서**
- [GAN 논문 이해하기](https://angrypark.github.io/First-GAN/)
- [GAN Code 분석하기](https://angrypark.github.io/GAN-tutorial-1/)
- [Conditional-GAN 논문 이해하기](https://kangbk0120.github.io/articles/2017-08/conditional-gan)
- [Conditional-GAN으로 내 핸드폰 손글씨 만들기](https://kangbk0120.github.io/articles/2017-08/condgan-imple)
- [DCGAN 논문 이해하기](https://angrypark.github.io/DCGAN-paper-reading/)
- [DCGAN pytorch로 구현하기](https://kangbk0120.github.io/articles/2017-08/dcgan-pytorch)
- [InfoGAN 논문 이해하기](https://kangbk0120.github.io/articles/2017-08/info-gan)
- [InfoGAN Code 분석하기](https://github.com/YBIGTA/Deep_learning/blob/master/GAN/2017-09-02-InfoGAN-implementation.ipynb)
- [DiscoGAN 논문 이해하기](https://angrypark.github.io/DiscoGAN-paper-reading/)

---


**목차**

- [0. Abstract](#0-abstract)
- [1. Introduction](#1-introduction)
- [2. Model](#2-model)
  - [2.1. Formulation](#21-formulation)
  - [2.2. Notation and Architecture](#22-notation-and-architecture)
  - [2.3. GAN with a Reconstruction Loss](#23-gan-with-a-reconstruction-loss)
  - [2.4. Our Proposed Model : Discovery GAN](#24-our-proposed-model-discovery-gan)
- [Reference](#reference)

---
> 논문 pdf :
[Learning to Discover Cross-Domain Relations
with Generative Adversarial Networks, 2016](https://arxiv.org/pdf/1703.05192.pdf)

---

## 0. Abstract
지금까지는 GAN을 통해 사람이 구별하기 어려운 가짜 이미지를 만들어내는 데 목표를 두었고 어느정도 성공하였습니다. 그러나 우리가 원하는 목적의 가짜 이미지를 만들기 위해서는 그 이상의 목표가 해결되어야 합니다. 예를 들면, 소 그림을 넣으면 비슷한 느낌의 말을 만들어준다던가, 여러분의 사진을 넣으면 같은 사진인데 성별만 바뀌는 이미지를 만들어준다던가 말이죠. 이를 하나의 도메인에서 mode는 유지한 채로 다른 도메인으로 바꿔주는 목표로 생각할 수 있는데, 오늘 소개할 논문에서는 이를 멋지게 구현해주었습니다. 이번 글에서는 문제를 어떻게 정의내리고, 어떤 구조와 컨셉을 통해 이를 해결하였는지 살펴보도록 하겠습니다.

---
## 1. Introduction
사람들은 두개의 다른 도메인이 주어졌을 때 그 관계를 쉽게 찾아냅니다. 예를 들어, 영어로 구성된 문장을 프랑스어로 번역하여 주어진다면, 그 두 문장의 관계를 사람들은 쉽게 찾아낼 수 있습니다 (의미는 같다, 언어는 다르다). 또한, 우리는 우리가 입고 있는 정장과 비슷한 스타일을 가지고 있는 바지와 신발을 쉽게 찾아낼 수 있습니다. 같은 스타일을 가지되 도메인만 정장에서 바지나 신발로 옮겨주는 것이죠.

그럼, 과연 기계도 서로 다른 두 개의 도메인 이미지의 관계를 찾아낼 수 있을까요? 이 문제는 *“한 이미지를 다른 조건이 달려 있는 이미지로 재생성할 수 있을까?”* 라는 문제로 재정의됩니다. 다시 말하면, 같은 이미지인데 한 도메인에서 다른 도메인으로 mapping해주는 함수를 찾을 수 있는가의 문제인 것이죠. 사실 이 문제는 최근 엄청난 관심을 받고 있는 GAN에서 이미 어느 정도 해결되었습니다. 그러나 GAN의 한계는 사람이나 다른 알고리즘이 직접 명시적으로 짝지은 데이터를 통해서만 문제를 해결할 수 있다는 것입니다. (예, 국방무늬를 갖고 있는 옷을 바꾸면 국방무늬를 가진 신발이 돼!)

명시적으로 라벨링된 데이터는 쉽게 구해지지 않으며, 많은 노동력과 시간을 필요로 합니다. 더군다나, 짝 중 하나의 도메인에서라도 그 사진이 없는 경우 문제가 생기고, 쉽게 짝짓기 힘들 정도로 훌륭한 선택지가 다수 발생하기도 하죠. 따라서, 이 논문에서는 우리는 2대의 다른 도메인에서 그 어떠한 explicitly pairing 없이 관계를 발견하는 것을 목표로 합니다(관계를 '발견'한다고 해서 DiscoGAN입니다.)

이 문제를 해결하기 위해서, 저자는 **Discover cross-domain relations with GANs** 을 새롭게 제안하였습니다. 이전에 다른 모델들과 달리, 우리는 그 어떤 라벨도 없는 두 개의 도메인 데이터 셋을 pre-training없이 train합니다.(이하 $A$,$B$ 도메인이라 명시할께요). Generator는 도메인 $A$의 한 이미지를 input으로 해서 도메인 $B$으로 바꿔줍니다. DiscoGAN의 핵심은 두개의 서로 다른 GAN이 짝지어져 있다는 것인데, 각각은 $A$를 $B$로, $B$를 $A$로 바꿔주는 역할을 해줍니다. 이 때의 핵심적인 전제는 하나의 도메인에 있는 모든 이미지를 다른 도메인의 이미지로 표현할 수 있다는 것입니다.

![1](assets/images/2017-09-09-DiscoGAN-paper-reading/1.png)

 결론부터 이야기 하자면 DiscoGAN은 Toy domain 과 real world image dataset에서 다 cross-domain relations를 알아내는 데 적합하다는 것을 확인할 수 있었습니다. 단순한 2차원 도메인에서 얼굴이미지 도메인으로 갔을 때에도 DiscoGAN 모델은 mode collapse problem에 좀더 robust하다는 것도 확인할 수 있었죠. 또한 얼굴, 차, 의자, 모서리, 사진 사이의 쌍방향 mapping에도 좋은 이미지 전환 성능을 보여주었습니다. 전환된 이미지는 머리 색, 성별, orientation 같은 특정한 부분만 바뀔수도 있었습니다.

---
## 2. Model
우리는 DiscoGAN이 어떤 문제들을 해결할 수 있는지 알아보았습니다. 이제 이 모델이 어떻게 이 문제를 해결하는지 좀 더 자세히 분석해보죠.

### 2.1. Formulation
관계라는 것은 $G_{AB}$로 정의내려질 수 있습니다. 즉 $G_{AB}$라는 것은 도메인 $A$에 있는 성분들을 $B$로 바꿔주는 것을 의미합니다. 완전 비지도 학습에서는, $G_{AB}$와 $G_{BA}$는 모두 처음에 정의내릴 수 없습니다. 따라서, 일단 모든 관계는 1대1 대응으로 만들어주고 시작합니다. 그러면 자연스럽게, 각각의 대응은 $G_{AB}$가 되며, $G_{BA}$는 $G_{AB}$의 역반응이 됩니다.

함수 $G_{AB}$의 범위는 도메인 $A$에 있는 모든 $x_A$가 도메인 $B$에 있는 $G_{AB}(x_A)$로 연결된 것입니다.

자 이를 목적함수로 표현해봅시다. 이상적으로는, 보시는 것처럼 $G_{BA} \circ G_{AB}(x_A) = x_A$이면 됩니다. 하지만 이런 제한식은 너무 엄격해서 이를 만족시키기 어렵습니다(사실 불가능하죠. generate해서 원래 사진 그대로 나온다는게 ㅎㅎ).  따라서 여기서는 $d(G_{BA} \circ G_{AB}(x_A), x_A)$를 최소화하려고 합니다. 비슷하게, $d(G_{AB} \circ G_{AB}(x_B), x_B)$도 최소화해야합니다. 이를 Discriminator와 generative adversarial loss가 들어간 loss로 표현하면 다음과 같습니다.

### 2.2. Notation and Architecture
각각의 Generator와 Discriminator의 input, output 형태는 다음과 같습니다.

ightarrow \mathbb{R}_{B}^{64	imes64	imes3}$

ightarrow [0,1]$b{R}_{A}^{64	imes64	imes3}

입니다.

### 2.3. GAN with a Reconstruction Loss
![2](assets/images/2017-09-09-DiscoGAN-paper-reading/2.png)

처음에는 기존 GAN을 약간 변형한 구조를 생각했었다고 합니다(그림 2-a). 기존 GAN은 input이 gaussian noise였던 것 기억하시나요? 여기서는 일단 input을 도메인 $A$의 image로 해줍니다. 이를 기반으로 generator가 fake image를 만들어내고, 이를 Discriminator는 도메인 $B$의 이미지와 함께 넣어서 무엇이 진짜인지를 구분하게 합니다. 즉, Generator의 입장에서는 비록 input은 도메인 $A$였지만, Discriminator를 속이기 위해서는 도메인 $B$와 유사한 이미지를 만들어야 한다는 것이죠. 이렇게만 잘 학습이된다면, Generator는 앞서 $G_{AB}$의 역할을 충실히 할 수 있게 됩니다. 도메인 $A$를, 도메인 $B$로 바꿔주는 역할을 해주는 것이죠.

![3](assets/images/2017-09-09-DiscoGAN-paper-reading/3.png)

하지만 이는 $A$에서 $B$로 가는 mapping만 배우게 됩니다. 동시에 $B$에서 다시 $A$로 가는 mapping도 학습하기 위해서 그림 2-b에서처럼 두번째 generator를 추가하게 됩니다. 또한 reconstruction loss도 추가하는데요, 이러한 과정들을 통해, 각각의 generator는 input 도메인에서 output 도메인으로 mapping하는 것은 물론 그 관계까지 discover하게 됩니다. 이 때 각각의 함수를 정의하고 loss function을 정의하면 다음과 같습니다.

 $x_{AB} = G_{AB}(x_A)$

 $x_{ABA} = G_{BA}(x_{AB}) = G_{BA} \circ G_{AB}(x_A)$

 $L_{CONST_A} = d(G_{BA}\circ G_{AB}(x_A), x_A)$

 $L_{GAN_A} = - \mathbb{E}_{x_A \sim P_A}[logD_B(G_{AB}(x_A))] $

$L_{G_{AB}} = L_{GAN_{B}} + L_{CONST_{A}} $

$L_{D_B} = - \mathbb{E}_{x_B \sim P_B}[logD_B(D_{B}(x_B))] - \mathbb{E}_{x_A \sim P_A}[log(1 - D_B(G_{AB}(x_A))]$


### 2.4. Our Proposed Model : Discovery GAN
![4](assets/images/2017-09-09-DiscoGAN-paper-reading/4.png)

최종적으로 이 논문에서 구현한 모델은 앞서 언급했던 그림 2-b의 모델 2개를 서로 다른 방향으로 이어주는 것입니다(그림 2-c). 각각의 모델은 하나의 도메인에서 다른 도메인으로 학습하며, 각각은 reconstruction을 통해 그 관계도 학습하게 됩니다.($G_{ABA}$의 BA와 $G_{BAB}$의 BA는 다릅니다.) 이 때 $G_{AB}$의 두 개의 Generator와 $G_{BA}$의 두 개의 Generator는 서로 파라미터를 공유합니다. 그리고 $x_{BA}$와 $x_{AB}$는 각각 $L_{D_A}$, $L_{D_B}$로 들어가게 됩니다. 이전 모델과 중요한 차이는 두 도메인의 input 이미지가 다 reconstruct되었으며 그에 따라 두 개의 reconstruction loss($L_{CONST_A}$, $L_{CONST_B}$)가 생성된다는 것입니다.

이처럼 두 개의 모델을 짝지어줌으로서 전체 Generator의 loss는 다음과 같이 정의합니다.

egin{matrix}
L_G &=& L_{G_{AB}} + L_{G_{AB}} \
    &=& L_{GAN_B} + L_{CONST_A} + L_{GAN_A} + L_{CONST_B}
\end{matrix}

비슷하게 전체 Discriminator의 loss는 다음과 같이 정의합니다.

$L_D = L_{D_A} + L_{D_B}$

여기까지 DiscoGAN의 성능, 발전 과정, 해결할 수 있는 문제들, 그리고 각각의 구조와 loss function을 알아보았습니다. 다음에는 코드로, 몇몇 실험에 대해 어떻게 문제를 정의 내리고 해결했는지 알아보겠습니다. 아디오스~

---
## Reference

[Learning to Discover Cross-Domain Relations with Generative Adversarial Networks, 2016](https://arxiv.org/pdf/1703.05192.pdf)

[sweetzLab 기술블로그 - DiscoGan (Learning to Discover Cross-Domain Relations with Generative Adversarial Network)](http://dogfoottech.tistory.com/170)

---
