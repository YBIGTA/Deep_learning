
# BEGAN: Boundary Equilibrium Generative Adversarial Networks



### 0. 들어가며

지난 시간, 우리는 DiscoGAN 논문 리뷰와 GAN을 통한 Colorization 구현에 대해 살펴보았습니다. 오늘 살펴볼 모델은 BEGAN 입니다. BEGAN은 올해 3월 구글이 발표한 논문, BEGAN: Boundary Equilibrium Generative Adversarial Networks에서 제안된 모델입니다. 최근 GAN팀이 살펴본 내용이 원 데이터의 transfer에 가까운 내용이었다면, 오늘 살펴볼 BEGAN은 고품질의 데이터 생성과 안정적인 학습에 초점이 맞춰져있는 모델입니다. 결과물부터 공유하자면 아래와 같습니다.
<div style="text-align:center"><figure><img src="https://4.bp.blogspot.com/-QfgccPH6_m0/WONBDxjjtJI/AAAAAAAABh0/lpxaxGzZcwYKKqraNfTweTplKvL0zlSsgCK4B/s1600/began_1.PNG"></img></figure></div>

> BEGAN으로 생성된 이미지

<br></br>
비교를 위해 다른 GAN 모델의 결과물과 비교하자면 아래와 같습니다.
<br></br>
<div style="text-align:center"><figure><img src="https://raw.githubusercontent.com/carpedm20/DCGAN-tensorflow/master/assets/result_16_01_04_.png"><figcaption></figcaption></img></figure></div>
<font color="gray"></font>

> DCGAN으로 생성된 이미지

<br></br>
<div style="text-align:center"><figure><img src="http://1.bp.blogspot.com/-QfuB07sp1I4/WONC6coa43I/AAAAAAAABig/hdFEweUOaEcKRsMaiEpllOmSdGQ0VZIGwCK4B/s1600/began_1_1.PNG"><figcaption><font color="gray"></font></figcaption></img></figure></div style="text-align:center">

> EBGAN 결과물과의 비교

<br></br>

생성되는 이미지 퀄리티 자체가 비약적으로 발전했다고 느껴지는데요, 본격적으로 BEGAN의 논문에 대해 살펴보겠습니다.

---
### 1. Introduction
우선 GAN에 대해 간략하게 복습해봅시다. GAN은 두 가지 네트워크로 구성됩니다.

1. 실제 데이터와 생성된 가짜 데이터를 구별하는 Discriminator,
2. Discriminator가 제대로 작동하지 않도록 실제 데이터와 유사한 가짜 데이터를 생성하는 Generator.

이 두 가지 뉴럴 네트워크의 경쟁적인 학습을 기반으로 실제와 유사한 데이터를 생성하는 것이 GAN의 핵심이지요. 이 논문의 Contribution은 아래와 같습니다.

>* 간단하지만 로버스트한 구조를 가지며, 빠르지만 안정적으로 학습시킬 수 있는 GAN 구조 제안
>* Discriminator와 Generator 사이 균형을 맞춰주는 Equilibrium concept
>* 생성 이미지의 다양성과 퀄리티 사이의 trade-off에 대한 control
>* approximate measure of convergence

---

### 2. Related Work

BEGAN이 기반으로 하고 있는 선행 연구들은 아래와 같습니다.

1.DCGAN - Deep Convolutional GAN
이미지 생성의 퀄리티를 높인 모델입니다. MNIST가 아닌 다른 이미지셋에 대해서 본격적으로 적용하기 시작한 모델이기도 하구요. Convolutional Architecture 도입을 통해 보다 안정적인 GAN 학습을 가능케 했습니다.
<br></br>
2.EBGAN - Energy Based GAN
앞서 Discriminator가 실제 이미지와 가짜 이미지를 분류하는 역할을 한다고 했는데요. EBGAN은 이 개념을 약간 틀어 D(x)를 에너지 함수로 적용했다고 합니다. 사실 저도 EBGAN 논문은 읽어보지 않아 자세한 내용은 모르겠지만, Discriminator를 Classifier가 아니라 AutoEncoder 방식으로 구현하기 시작한 모델로 알고 있습니다. BEGAN 또한 이런 방식을 채택하고 있으며, 자세한 내용은 뒤에 적어보겠습니다.
<br></br>
3.WGAN - Wasserstein GAN
지난 2016년 페이스북에서 낸 논문입니다. 학습의 안정성을 높이는 데 기여했으며, mode collapse를 막는 방법을 고안했다고 합니다. 역시 아직 읽어보지는 못했지만, 여기서 고안한 loss가 모델 수렴에 대한 척도로서도 사용이 된다고 합니다. 논문 초록에 <font color="gray">provide meaningful learning curves useful for debugging and hyperparameter searches</font> 라고 적혀있군요. BEGAN은 이러한 WGAN의 모델 수렴 척도를 이용/발전 시켰습니다.

---

### 3 Proposed method

#### 3.1 Wasserstein distance lower bound for auto-encoders

기존 GAN의 학습 방식은 아래와 같습니다.
1. Generator를 통해 fake sample을 만들어낸다.
2. Discriminator가 fake sample과 real sample을 잘 구별하게 학습시킨다.
3. Generator가 Discriminator가 실수를 하도록 (fake sample을 보고 real sample로 판별하게 하도록) 학습을 시킨다.

결국엔 Discriminator가 fake sample과 real sample을 구별하는, 일종의 분류기 역할을 합니다. 이를 통해서 생성된 가짜 데이터의 분포를 실제 데이터의 분포에 근사시키는 거죠. 실제 데이터의 분포를 학습하는겁니다.

BEGAN은 다른 학습 방식을 사용합니다. data distribution이 아닌 loss distribution에 더 집중합니다. 여기서 데이터의 loss를 산출하기 위해 AutoEncoder가 도입되는거죠. 그래서 BEGAN의 Discriminator의 작동 방식은 아래와 같습니다.

1. 실제 이미지를 Discriminator(AutoEncoder)에 집어넣는다.
2. 실제 이미지에 대한 복원 이미지가 결과물로 나온다.
3. 복원 이미지의 pixel-wise reconstruction loss를 구한다.
4. 가짜 이미지에 대해서 1-3번 과정을 거친다. 그러면 우리는 실제 이미지와 가짜 이미지에 대한 reconstruction loss를 갖게 된다.


reconstruction loss를 구하는 방법을 수식으로 표현하자면 아래와 같습니다.
<div style="text-align:center"><figure><img src="https://1.bp.blogspot.com/-d2_e4HkrZY4/WORHCWKxTaI/AAAAAAAABjg/deUJ9j3EqWUN-WBZZVX13czb6kAaV2VewCK4B/s1600/began_7.PNG"></img></figure></div>
여기서 <a href="https://www.codecogs.com/eqnedit.php?latex=\eta" target="blank"><img src="https://latex.codecogs.com/gif.latex?\eta" title="\eta" /></a>값이 1이면 L1 loss, 2가 되면 L2 loss가 될텐데요. BEGAN에서는 L1 loss를 이용했습니다. 여기서 산출한 실제 이미지의 loss 분포를 <a href="https://www.codecogs.com/eqnedit.php?latex=\mu_1" target="blank"><img src="https://latex.codecogs.com/gif.latex?\mu_1" title="\mu_1" /></a>, fake 이미지의 loss 분포를 <a href="https://www.codecogs.com/eqnedit.php?latex=\mu_2" target="blank"><img src="https://latex.codecogs.com/gif.latex?\mu_2" title="\mu_2" /></a>라고 합시다. 그럼 두 분포 사이의 거리를 산출하고, 이를 기반으로 수렴을 시킬수가 있어요. 여기에는 KL등 여러가지 방식을 사용할 수 있는데, 여기서는 Wasserstein Distance가 사용되었습니다. Wasserstein Distance는 KL에 비해 약한(weak) 수렴을 판정하는 데 무른(soft) 성질을 갖는다고 합니다(...) 분포 수렴을 판단하는 기준이 덜 빡세다(?)는 수준으로 이해했습니다. 무튼 distance를 구하는 식은 아래와 같습니다.
<br></br>
<div style="text-align:center"><figure><a href="https://imgur.com/WwuDE4C"><img src="https://i.imgur.com/WwuDE4C.png" title="source: imgur.com" /></img></a></figure></div>

<a href="https://www.codecogs.com/eqnedit.php?latex=\Gamma" target="blank"><img src="https://latex.codecogs.com/gif.latex?\Gamma" title="\Gamma" /></a>는 두 확률분포 <a href="https://www.codecogs.com/eqnedit.php?latex=\mu_1" target="blank"><img src="https://latex.codecogs.com/gif.latex?\mu_1" title="\mu_1" /></a>과 <a href="https://www.codecogs.com/eqnedit.php?latex=\mu_2" target="blank"><img src="https://latex.codecogs.com/gif.latex?\mu_2" title="\mu_2" /></a>의 모든 결합확률분포입니다. inf는 the greatest lower bound(하한 중 가장 큰 값)을 의미하구요. 결국  Wasserstein Distance는 이 결합확률분포들 중에서 distance의 기댓값을 가장 작게 추정한 값을 의미합니다. 여기에 distance의 개념으로 L1 distance가 사용된거죠. 무튼 여기에 Jensen 부등식을 적용해서 distance의 하한을 구하면 이렇게 됩니다.

<div style="text-align:center"><figure><a href="https://imgur.com/42X9neM"><img src="https://i.imgur.com/42X9neM.png" title="source: imgur.com" /></a></figure></div>

이래저래 복잡하게 설명했지만 결론은 실제 이미지와 fake 이미지의 reconstruction loss를 구해서 각각의 평균 차를 구하면 되는거였네요.


AutoEncoder의 개념에 대해서는 [이활석 박사님의 발표 자료](https://mega.nz/#!tBo3zAKR!yE6tZ0g-GyUyizDf7uglDk2_ahP-zj5trVZSLW3GAjw)를, Wasserstein Distance가 궁금하신 분들은 [Wasserstein GAN 수학 이해하기](https://www.slideshare.net/ssuser7e10e4/wasserstein-gan-i)를 참고하시면 좋을 것 같습니다.
<br></br>

#### 3.2 GAN objective

먼 길 걸어왔습니다. 근데 이걸 GAN 목적함수로 어떻게 표현할 수 있을까요..?? 결론부터 말씀드리자면 아까 구한 Wasserstein distance(W)를 maximizing하는 방향으로 진행됩니다. W를 극대화시키는 방법은 아래 두 가지 입니다.

<div style="text-align:center"><figure><img src="http://1.bp.blogspot.com/-CUitaK88Z7U/WORLPnbwuHI/AAAAAAAABkE/0CDOEBBKXAQ_05uFMNfLdfJAr-ItbDubQCK4B/s1600/began_9.PNG"></img></figure></div>

BEGAN은 여기서 (b) 방식을 사용합니다. 잘 생각해봅시다. 여기서 <a href="https://www.codecogs.com/eqnedit.php?latex=m_1" target="blank"><img src="https://latex.codecogs.com/gif.latex?m_1" title="m_1" /></a>은 AutoEncoder에 실제 이미지를 넣었을 때 reconstruction loss였어요. AutoEncoder가 실제 이미지를 잘 복원시키는 방향으로, 즉 loss를 minimize하는 방향으로 학습하는 게 맞겠죠? 이제 <a href="https://www.codecogs.com/eqnedit.php?latex=m_2" target="blank"><img src="https://latex.codecogs.com/gif.latex?m_2" title="m_2" /></a>를 살펴봅시다. <a href="https://www.codecogs.com/eqnedit.php?latex=m_2" target="blank"><img src="https://latex.codecogs.com/gif.latex?m_2" title="m_2" /></a>는 가짜 이미지에 대한 reconstruction loss에요. 가짜 이미지는 generator가 만드는 이미지입니다. Random noise에서 이미지를 만들기 때문에 학습 초기에는 가짜 이미지가 그냥 까만 이미지밖에 나올 수밖에 없어요.<a href="https://www.codecogs.com/eqnedit.php?latex=m_2" target="blank"><img src="https://latex.codecogs.com/gif.latex?m_2" title="m_2" /></a>를 maximizing하는 과정은 가짜 이미지를 보다 복잡한 형태로 만드는 과정이고, 결국에는 실제 이미지와 유사한 가짜 이미지를 만들어내려는 GAN의 과정이라고 생각됩니다.

다시... 이제 Discriminator와 Generator의 목적함수를 보다 구체적으로 표현하자면 아래와 같습니다.

<div style="text-align:center"><figure><a href="https://imgur.com/UlFha9G"><img src="https://i.imgur.com/UlFha9G.png" title="source: imgur.com" /></img></a></figure></div>

Discriminator는 실제 이미지와 가짜 이미지간의 reconstruction loss 차를 minimizing하는 거구요. Generator는 반대로 이걸 maximizing하는겁니다.
<br></br>

#### 3.3 Equilibrium

위 목적함수를 그대로 대입했을 경우 GAN의 균형은 아래와 같이 나와요.
<figure><a href="https://imgur.com/GOYBt55"><img src="https://i.imgur.com/GOYBt55.png" title="source: imgur.com" /></a></figure>

기존 GAN이랑 다를 게 없죠. 근데 앞선 발제들을 통해서 꾸준히 지적되고 있는 문제는 이 이론상의 균형이 실제 적용단에 있어서 그닥 좋지 못하다는 거에요. 그래서 BEGAN에서는 두 Expectation 사이에 균형을 조정합니다.
<div style="text-align:center"><figure><a href="https://imgur.com/cfeFZms"><img src="https://i.imgur.com/cfeFZms.png" title="source: imgur.com" /></a></figure></div>

여기서 <a href="https://www.codecogs.com/eqnedit.php?latex=\gamma" target="blank"><img src="https://latex.codecogs.com/gif.latex?\gamma" title="\gamma" /></a>를 diversity ratio라고 부릅니다. 즉, 감마값이 높아지면 generator에 조금 더 힘이 실리고, 따라서 결과 이미지가 보다 다양해지는거죠. 감마값이 낮아지면 discriminator, 즉 오토인코더에 힘이 실려서 이미지를 복원하는 데 집중하게 됩니다. 따라서 이미지 퀄리티가 높아지게 되구요.

#### 3.4 Boundary Equilibrium GAN

자, 그래서 이 diversity ratio까지 고려된 BEGAN의 목적함수는 어떻게 될까요?
<div style="text-align:center"><figure><a href="https://imgur.com/BF8c6bf"><img src="https://i.imgur.com/BF8c6bf.png" title="source: imgur.com" /></img></a></figure></div>

자, 그래서 이 diversity ratio까지 고려된 BEGAN의 목적함수는 어떻게 될까요? 위에 조정된 균형을 만족시키기 위해서 Proportional Control Theory를 사용했다고 합니다. Generator에 얼마나 많은 힘을 실어줄 지 <a href="https://www.codecogs.com/eqnedit.php?latex=\lambda_k" target="blank"><img src="https://latex.codecogs.com/gif.latex?\lambda_k" title="\lambda_k" /></a>는 <a href="https://www.codecogs.com/eqnedit.php?latex=k_t" target="blank"><img src="https://latex.codecogs.com/gif.latex?k_t" title="k_t" /></a> 를 통해 조정하구요. 이 <a href="https://www.codecogs.com/eqnedit.php?latex=k_t" target="blank"><img src="https://latex.codecogs.com/gif.latex?k_t" title="k_t" /></a> 는 위 식처럼 업데이트 됩니다. <a href="https://www.codecogs.com/eqnedit.php?latex=\lambda_k" target="blank"><img src="https://latex.codecogs.com/gif.latex?\lambda_k" title="\lambda_k" /></a>는 <a href="https://www.codecogs.com/eqnedit.php?latex=k_t" target="blank"><img src="https://latex.codecogs.com/gif.latex?k_t" title="k_t" /></a> 조정에 대한 learning rate라고 생각하면 될 것 같습니다. 학습 진행과정 내내 Discriminator의 loss가 Generator의 loss보다 큰 것을 확인했다고 합니다. 왜냐하면 학습 초기 생성된 데이터가 0에 가까울 테니 복원하기가 훨씬 쉽기 떄문이겠죠.

Discriminator와 Generator를 도식화하자면 아래와 같습니다. Discriminator는 encoder와 decoder로 구성이 되어있고, Generator는 이 decoder와 같은 구조를 취합니다.
<div style="text-align:center"><figure><img src="https://2.bp.blogspot.com/-DCnXnBGg7G8/WOXMjR3pDhI/AAAAAAAABlk/aWqQPBdlaRcT2u2VkrZBrtHSBn76DTPbgCK4B/s1600/began_12.PNG"></img></figure></div>


#### 3.4 Boundary Equilibrium GAN
GAN에서는 두 가지 loss가 사용됩니다. 기본적으로 제로섬게임을 기반으로 하기 때문에 한 쪽 loss는 0으로 수렴하고 다른 loss는 높아지는 경향을 보이죠. 그럼 우리는 무엇을 기준으로 GAN이 제대로 수렴하고 있구나를 판단할 수 있을까요? 여기에 대해서 BEGAN 논문에서는 Global Convergence Measurement를 제안합니다.

<div style="text-align:center"><figure><a href="https://imgur.com/voezeEn"><img src="https://i.imgur.com/voezeEn.png" title="source: imgur.com" /></a></figure></div>
이 값을 통해서 모델이 최종적으로 수렴했는 지, 혹은 중간에 mode collapse가 발생했는 지 판단할 수 있습니다.


---

### 4. Experiments
#### 4.1 Setup
- 기본적으로 128x128 이미지 생성
- hidden layer 사이즈는 64
- noise vector 사이즈도 64
- batchSize = 16
- Training time was about 2.5 days on four P100 GPUs. (....)
- 32 x 32 에 대해서는 GPU 1개로도 수시간 내 학습 가능

#### 4.2 Image diversity and quality
<div style="text-align:center"><figure><img src="https://i.imgur.com/0PtaFSV.png" title="source: imgur.com" /></img></figure></div>

#### 4.3 Space continuity
<div style="text-align:center"><figure><a href="https://imgur.com/uxMv01D"><img src="https://i.imgur.com/uxMv01D.png" title="source: imgur.com" /></a></figure></div>

#### 4.4 Convergence measure and image quality
<div style="text-align:center"><figure><a href="https://imgur.com/B4aOrMi"><img src="https://i.imgur.com/B4aOrMi.png" title="source: imgur.com" /></a></figure></div>

---

### Epilogue: 구현

구현중인 코드를 [GAN팀 깃허브](https://github.com/YBIGTA/Deep_learning/blob/master/GAN/2017-09-23-BEGAN-implementation.py)에 올려두었습니다.

개인적으로 InfoGAN에 비해서 코드 구현 자체는 수월했습니다.(물론 [이활석님의 텐서플로 구현](https://github.com/hwalsuklee/tensorflow-generative-model-collections)과 [김태훈님의 파이토치 구현](https://github.com/carpedm20/BEGAN-pytorch)을 참고했기 때문에...) Parameter를 공유한다든지 Pytorch에서 구현하기 불편한 테크닉들이 적기도 했습니다. 물론, 학습은 실패했습니다... 데이터는 지난 번에 사용했던 한국 여자 연예인 이미지 데이터를 사용했는데요. 아래는 15000 step 학습했을 때 이미지입니다.

<div style="text-align:center"><figure><a href="https://imgur.com/kt9ODMC"><img src="https://i.imgur.com/kt9ODMC.png" title="source: imgur.com" /></a></figure></div>

얼굴이 뭉개질때마다 학습율을 조정해보았으나 최대한 얼굴 형태와 비슷하게 나온 게 이 정도였습니다. 계속 같은 방법으로 학습시키면 이미지가 생성될까 싶기도 했지만 너무 time-consuming한 작업이라 포기했습니다.. 예상되는 문제는 (1) 데이터 셋이  (2)구현 오류 둘 중 하나인데, (2)번 문제면 조금 슬플 것 같습니다.. 다음 주에 발제는 못하더라도 celebA 데이터에 한 번 적용해보고 결과를 공유해보겠습니다.

##### 17.09.29 update
<a href="https://imgur.com/tKFk1mq"><img src="https://i.imgur.com/tKFk1mq.png" title="source: imgur.com" /></a>
> 약 30000 step 학습한 결과

---

##### Reference
1. [초짜 대학원생의 입장에서 이해하는 BEGAN: Boundary Equilibrium Generative Adversarial Networks](http://jaejunyoo.blogspot.com/2017/04/began-boundary-equilibrium-gan-1.html)

2. [BEGAN: Boundary Equilibrium Generative Adversarial Networks](https://www.google.co.kr/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0ahUKEwjtwb_Xk7rWAhVKQLwKHScnCywQFgglMAA&url=https%3A%2F%2Farxiv.org%2Fabs%2F1703.10717&usg=AFQjCNGbbM7fAFt7CHTHIU9N8pRFXoSQ-A)
