---
title: "[GAN] DCGAN 논문 이해하기"
layout: post
date: 2017-08-03 20:30
image: /assets/images/2017-08-03-DCGAN-paper-reading/background.png
headerImage: true
tag:
- gan
- pytorch
- deep learning
- ybigta
- dcgan
category: blog
author: angrypark
description: DCGAN 논문 처음부터 끝까지 차근차근 이해하기
# jemoji: '<img class="emoji" title=":ramen:" alt=":ramen:" src="https://assets.github.com/images/icons/emoji/unicode/1f35c.png" height="20" width="20" align="absmiddle">'
---

<span style="color:#7C7877; font-family: 'Apple SD Gothic Neo'; font-weight:200">

## 요약

지금까지 가장 기본적인 GAN의 이론적 내용과 그 코드의 작동 방법에 대해 살펴보았다면, 이제는 GAN을 활용한 다양한 논문들로 그 이론을 확장하고자 한다. 셀 수도 없을 만큼 다양한 GAN 응용 논문들이 있지만, 가장 기본적인 응용 GAN 논문으로 시작한다. 바로 2016년 발표된 **Deep Convolutional Generative Adversarial Nets (DCGAN)** 이다.

> 논문 pdf :
[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks - Alec Radford el ec, 2016](https://arxiv.org/abs/1511.06434)

**목차**

- [GAN Review](#gan-review)
- [기존 GAN의 한계](#기존-gan의-한계)
- [Architecture Guidelines](#architecture-guidelines)
- [Generator Model](#generator-model)
- [Visualization](#visualization)
- [Reference](#reference)

---
## GAN Review
![gan-workflow](/assets/images/2017-08-03-DCGAN-paper-reading/gan-workflow.jpg)

> [[GAN] First GAN](https://angrypark.github.io/First-GAN/)
[[GAN] 1D Gaussian Distribution Generation](https://angrypark.github.io/GAN-tutorial-1/)

---

## 기존 GAN의 한계
기존 GAN의 한계는 다음과 같다.

#### 1. GAN은 결과가 불안정하다
기존 GAN만 가지고는 좋은 성능이 잘 안나왔다.

#### 2. Black-box method
Neural Network 자체의 한계라고 볼 수 있는데, 결정 변수나 주요 변수를 알 수 있는 다수의 머신러닝 기법들과 달리 Neural Network은 처음부터 끝까지 어떤 형태로 그러한 결과가 나오게 되었는지 그 과정을 알 수 없다.

#### 3. Generative Model 평가
GAN은 결과물 자체가 **새롭게 만들어진 Sample** 이다. 이를 기존 sample과 비교하여 얼마나 비슷한 지 확인할 수 있는 정량적 척도가 없고, 사람이 판단하더라도 이는 주관적 기준이기 때문에 얼마나 정확한지, 혹은 뛰어난지 판단하기 힘들다.

---
## DCGAN의 목표
#### Generator가 단순 기억으로 generate하지 않는다는 것을 보여줘야 한다.

#### z의 미세한 변동에 따른 generate결과가 연속적으로 부드럽게 이루어져야 한다.

---
## Architecture Guidelines
GAN과 DCGAN의 전체적인 구조는 거의 유사하다. 다만 각각의 Discriminator와 Generator의 세부적인 구조가 달라진다. 논문에서는 이 구조를 개발한 방법을 다음과 같이 소개한다.
>*After extensive model exploration we identified a family of architectures that resulted stable training across a range of datasets and allowed for training higher resolution and deeper generative models.*

쉽게 말하면 엄청난 노가다(?) 끝에 안정적이고 더 성능이 향상된 결과를 찾게 되었다는 말이다.

#### 기존 GAN Architecture
기존 GAN은 자세히 살펴보면 다음과 같은 아주 간단하게 fully-connected로 연결되어 있다. ![gan-architecture](/assets/images/2017-08-03-DCGAN-paper-reading/gan-architecture.png)

#### CNN Architecture
CNN은 이러한 fully-connected 구조 대신에 convolution, pooling, padding을 활용하여 레이어를 구성한다.

![cnn-architecture](/assets/images/2017-08-03-DCGAN-paper-reading/cnn-architecture.png)

#### DCGAN Architecture
DCGAN은 결국, 기존 GAN에 존재했던 fully-connected구조의 대부분을 CNN 구조로 대체한 것인데, 앞서 언급했던 것처럼 엄청난 시도들 끝에 다음과 같이 구조를 결정하게 되었다.

![architecture-guidelines](/assets/images/2017-08-03-DCGAN-paper-reading/architecture-guidelines.png)

- Discriminator에서는 모든 pooling layers를 **strided convolutions** 로 바꾸고, Generator에서는 pooling layers를 **fractional-strided convolutions** 으로 바꾼다.

- Generator와 Discriminator에 batch-normalization을 사용한다. 논문에서는 이를 통해 deep generators의 초기 실패를 막는다고 하였다. 그러나 모든 layer에 다 적용하면 sample oscillation과 model instability의 문제가 발생하여 Generator output layer와 Discriminator input layer에는 적용하지 않았다고 한다.

- Fully-connected hidden layers를 삭제한다.

- Generator에서 모든 활성화 함수를 Relu를 쓰되, 마지막 결과에서만 Tanh를 사용한다.

- Discriminator에서는 모든 활성화 함수를 LeakyRelu를 쓴다.

> **Strided convolutions?**
![padding-strides](/assets/images/2017-08-03-DCGAN-paper-reading/padding_strides.gif)

> **Fractionally-strided convolutions?**
![padding-strides-transposed](/assets/images/2017-08-03-DCGAN-paper-reading/padding_strides_transposed.gif)
논문을 읽으며 가장 이해가 안되었던 부분인데, 기존의 convolutions는 필터를 거치며 크기가 작아진 반면에, fractionally-strided convolutions은 input에 padding을 하고 convolution을 하면서 오히려 크기가 더 커지는 특징이 있다. 쉽게 transposed convolution이라고도 불리고, deconvolution이라고도 불리는데, deconvolution는 잘못된 단어라고 한다.

> **Batch-normalization?**
Batch Normalization은 2015년 arXiv에 발표된 후 ICML 2015 (마찬가지로 매우 권위 있는 머신러닝 학회)에 publish 된 이 논문 ([Batch Normalization : Accelerating Deep Network Training by Reducing Internal Covariance Shift](http://arxiv.org/abs/1502.03167)) 에 설명되어 있는 기법으로, 발표된 후 최근에는 거의 모든 인공신경망에 쓰이고 있는 기법이다. 기본적으로 Gradient Vanishing / Gradient Exploding 이 일어나지 않도록 하는 아이디어 중의 하나이며, 지금까지는 이 문제를 Activation 함수의 변화 (ReLU 등), Careful Initialization, small learning rate 등으로 해결하였지만, 이 논문에서는 이러한 간접적인 방법보다 training 하는 과정 자체를 전체적으로 안정화하여 학습 속도를 가속시킬 수 있는 근본적인 방법을 제안하였다.
더 자세한 내용은 다음 포스트를 참고하길 바란다.
[Batch Normalization 설명 및 구현](https://shuuki4.wordpress.com/2016/01/13/batch-normalization-%EC%84%A4%EB%AA%85-%EB%B0%8F-%EA%B5%AC%ED%98%84/)

---
## Generator Model
위에서 설명된 Generator의 구조를 시각화하면 다음과 같다.

![generator-model](/assets/images/2017-08-03-DCGAN-paper-reading/generator-model.png)

100 dimensional uniform distribution(Z)이 들어오면 이들이 4개의 fractionally-strided convolution layer을 거치며 크기를 키워서 더 높은 차원의 64x64 pixel 이미지가 된다.

---
## Visualization
#### Generated bedrooms
![Visualization](/assets/images/2017-08-03-DCGAN-paper-reading/visualization-1.png)

#### Walking in the latent space
![Visualization](/assets/images/2017-08-03-DCGAN-paper-reading/visualization-2.png)

앞서 DCGAN의 목표들 중 하나인 walking in the latent space를 직접 구현한 그림이다.

#### Visualize filters (no longer black-box)
![Visualization](/assets/images/2017-08-03-DCGAN-paper-reading/visualization-3.png)

#### Applying arithmetic in the input space
![Visualization](/assets/images/2017-08-03-DCGAN-paper-reading/visualization-4.png)

---
## Reference

---
