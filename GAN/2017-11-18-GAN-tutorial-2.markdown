---
title: "[GAN] 1D Gaussian Distribution Generation"
layout: post
date: 2017-11-18 15:10
image: /assets/images/170721-first-gan/background.png
headerImage: false
tag:
- gan
- pytorch
- deep learning
- ybigta
category: blog
author: angrypark
description: GAN으로 1D Gaussian Distribution Generate 하기
# jemoji: '<img class="emoji" title=":ramen:" alt=":ramen:" src="https://assets.github.com/images/icons/emoji/unicode/1f35c.png" height="20" width="20" align="absmiddle">'
---

<span style="color:#7C7877; font-family: 'Apple SD Gothic Neo'; font-weight:200">

## 요약

이 튜토리얼을 통해 기본적으로 GAN이 무엇인지 설명드리고, 이를 이용하여 1-d 가우시안 분포(정규분포)를 만들어보도록 하겠습니다.

**목차**

- [GAN?](#gan)
- [Pytorch 설치](#pytorch-설치)
- [Workflow](#workflow)
- [Algorithm](#algorithm)
- [Visualization](#visualization)
- [Reference](#reference)

---
## GAN?
GAN이 상당히 어려워 보이지만, 그 원리는 누구나 이해할 수 있을 정도로 간단합니다. GAN(Generative Adversarial Nets)이란 서로 대립하는 두 개의 신경망을 동시에 학습시키면서 **원본의 sample과 유사한 sample을 만들어내는 방식**을 말합니다.

여기서 가장 중요한 두 개의 역할이 바로 **Generator** 와 **Discriminator** 인데, 지폐 위조 범죄에 비추어 예시를 들자면, Generator는 실제 지폐와 똑같이 생긴 지폐를 만들려고 노력하는 **위조지폐범** 이고 Discriminator는 이를 제대로 구별하려는 **경찰** 이라고 볼 수 있습니다.

기존의 하나의 cost function을 가지고 이를 최적화 했던 다른 신경망 방식과 다르게, 두 개의 신경망을 동시에 학습시키면서 Generator는 Discriminator의 구분 확률을 최대한 줄이고, 이와 동시에 Discriminator는 real sample(실제 지폐)와 fake sample(위조 지폐)의 구분 정확도를 높이는 목적을 두고 있습니다.

NIPS2016에서 저자인 Ian Goodfellow가 발표한 내용에 따르면, GAN은 다음과 같은 상황에서 매우 뛰어난 성능을 보여준다고 합니다.


![Why GAN?](/assets/images/2017-07-23-GAN-tutorial-1/why-gan.png)


가장 핵심은, GAN은 기존의 머신러닝 기법과 같이 원데이터의 분포(모집단)에 대해 직접적으로 알아내고 분석한다기보다는, 원데이터와 최대한 비슷한 sample을 만들어내는 데에만 목표를 둔다는 점입니다. 따라서 원데이터가 매우 복잡하고 고차원이거나, 결측치가 많거나, 원데이터가 무한히 많을 때 좋은 성능을 보여줍니다. 그 예로, 원데이터를 파악하기 불가능한 예술의 분야를 들 수 있죠.

더 자세한 내용은 저번 포스트([Link](https://angrypark.github.io/First-GAN/))를 참고하시면 됩니다.

---
## Pytorch 설치
Windows의 경우 Anaconda를 설치한 상태에서는 다음의 명령어로 설치할 수 있습니다.
~~~
conda install -c peterjc123 pytorch=0.1.12
~~~
나머지(Mac OS, Linux)는 [공식 사이트](http://pytorch.org/)에서 다운로드 받으실 수 있습니다.

---
## Workflow
코드의 workflow는 다음과 같습니다.

![workflow](/assets/images/2017-07-23-GAN-tutorial-1/code-workflow.jpg)
#### Generator
먼저 Generator의 입장에서 말하자면, 입력값은 0과 1사이의 임의의 값이고, 이를 노이즈를 줘서 fake sample을 만들어 Discriminator에게 보내게 됩니다. 후에, Discriminator의 반응에 따라 신경망을 최적화하는데, 즉 어떤 sample에 대해서는 1(real)이라 반응했다 / 어떤 sample은 0(fake)이라 반응했다 의 여부에 따라 Generator가 최적화되는 것입니다. 인상깊은 점은 **Generator는 sample이 실제로 real sample인지 fake sample인지 모릅니다!** 다만 Discriminator가 어떻게 평가했는지의 여부만 보고 최적화하는 것입니다.

#### Discriminator
다음으로 Discriminator의 입장에서 말하지면, 입력값은 real sample와 fake sample이고 이들이 실제로 real인지 fake인지의 여부입니다(어떤 지폐와, 그 지폐가 위조인지 진짜인지의 여부). 일단 본인이 알아서 평가한 후에, 실제 real/fake 여부(정답)에 따라 본인을 최적화합니다. 이 때 본인이 평가한 정보를 Generator에게 보내게 됩니다(Generator도 최적화 할 수 있도록 말이죠).

---
## Algorithm

#### 필요한 라이브러리 불러오기
~~~python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import cufflinks as cf
import plotly
plotly.tools.set_credentials_file(username='your_username', api_key='your_api_key')
~~~

#### 파라미터 선정
~~~python
# 모집단의 평균, 표준편차 지정 : 실제 지폐의 모양
data_mean = 4
data_stddev = 1.25

# 모델의 파라미터 지정
g_input_size = 1     # Random noise dimension coming into generator, per output vector : (1,N)
g_hidden_size = 50   # G의 레이어 크기
g_output_size = 1    # G의 결과값의 크기 : (1,N) - 위조 지폐의 크기
d_input_size = 100   # Minibatch size - cardinality of distributions
d_hidden_size = 50   # D의 레이어 크기
d_output_size = 1    # 진짜인지 가짜인지의 여부(확률값) : (1,1)
minibatch_size = d_input_size

d_learning_rate = 2e-4  # 2e-4
g_learning_rate = 2e-4
optim_betas = (0.9, 0.999)
num_epochs = 2400 # 얼마나 최적화할지
print_interval = 10 # 몇번마다 결과를 출력할지
d_steps = 1  # 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator
g_steps = 1
~~~

#### 데이터 입력 관련 함수들
~~~python
# preprocess : data를 입력받고 preprocessing
(name, preprocess, d_input_func) = ("Data and variances", lambda data: decorate_with_diffs(data, 2.0), lambda x: x * 2)

print("Using data [%s]" % (name))

# ##### DATA: Target data and generator input data

# 한번에 바로 평균이 mu이고 표준편차가 sigma인 정규분포에서 샘플 (1,n)개 뽑기
def get_distribution_sampler(mu, sigma):
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))  # Gaussian

# 0부터 1사이의 숫자중 임의로 샘플 (m,n)개 뽑기
def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n)  # Uniform-dist data into generator, _NOT_ Gaussian
~~~

#### Generator & Discriminator

~~~python
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.sigmoid(self.map2(x))
        return self.map3(x)

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.elu(self.map2(x))
        return F.sigmoid(self.map3(x))
~~~

#### 기타 함수들
~~~python
# data의 정보를 리스트화하여 반환
def extract(v):
    return v.data.storage().tolist()

# data의 평균과 표준편차 반환
def stats(d):
    return [np.mean(d), np.std(d)]

# data를 입력받으면, data에 정규화한 data를 가로로 붙여서 반환
def decorate_with_diffs(data, exponent):
    # torch.mean(data,1) : (m,n)크기의 샘플을 각 행별로 평균을 구해서 (m,1)크기로 반환
    mean = torch.mean(data.data, 1)

    # torch.ones(m,n) : (m,n)크기의 1로만 이루어진 벡터
    # mean.tolist()[0][0] : data의 1행의 평균
    # mean_broadcast : data와 같은 크기의 벡터, 모든 성분은 data의 1행의 평균으로 구성됨
    mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])

    # diffs : dat에서 mean_broadcast를 뺀 후 이를 exponet제곱한다
    diffs = torch.pow(data - Variable(mean_broadcast), exponent)

    # torch.cat : concat의 의미, 0이면 세로로 붙이고, 1이면 가로로 붙임
    # return : data와 diff를 이어서 반환
    return torch.cat([data, diffs], 1)

# d_sampler : data의 평균과 표준편차와 같은 평균과 표준편차를 가지는 정규분포에서 sample뽑는 함수 : d_sampler(N)
d_sampler = get_distribution_sampler(data_mean, data_stddev)

# 임의의 0에서 1사이의 N개의 sample 뽑는 함수 : gi_sampler(N)
gi_sampler = get_generator_input_sampler()
~~~

#### G,D,optimizer 정의
~~~python
# G와 D 정의
G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
D = Discriminator(input_size=d_input_func(d_input_size), hidden_size=d_hidden_size, output_size=d_output_size)

criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss

# Optimizer 정의
d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)
~~~

#### 학습 시작
~~~python
for epoch in range(num_epochs):
    for d_index in range(d_steps):
        # 1. Train D on real+fake
        D.zero_grad()

        #  1A: 실제 데이터로 D 학습
        d_real_data = Variable(d_sampler(d_input_size))
        d_real_decision = D(preprocess(d_real_data))
        d_real_error = criterion(d_real_decision, Variable(torch.ones(1)))  # ones = true
        d_real_error.backward() # compute/store gradients, but don't change params

        #  1B: 가짜 데이터로 D 학습
        d_gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
        d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
        d_fake_decision = D(preprocess(d_fake_data.t()))
        d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(1)))  # zeros = fake
        d_fake_error.backward()
        d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

    for g_index in range(g_steps):

        # 2. D의 반응에 따라 G학습

        # G 초기화
        G.zero_grad()

        gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
        g_fake_data = G(gen_input)
        dg_fake_decision = D(preprocess(g_fake_data.t()))
        g_error = criterion(dg_fake_decision, Variable(torch.ones(1)))  # we want to fool, so pretend it's all genuine

        g_error.backward()
        g_optimizer.step()  # Only optimizes G's parameters

    if (epoch+1) % (print_interval*10) == 0:
        print("%s: D: %.4f/%.4f G: %.4f (Real: %s, Fake: %s) \n" % (epoch+1,
                                                            extract(d_real_error)[0],
                                                            extract(d_fake_error)[0],
                                                            extract(g_error)[0],
                                                            stats(extract(d_real_data)),
                                                            stats(extract(d_fake_data))))
    if (epoch+1) % print_interval == 0:
        [mu_real, sigma_real] = stats(extract(d_real_data))
        [mu_fake, sigma_fake] = stats(extract(d_fake_data))

        x1 = np.linspace(mu_real-9*sigma_real,mu+9*sigma_real, 100)
        x2 = np.linspace(mu_fake-9*sigma_fake,mu+9*sigma_fake, 100)
        plt.plot(x1,mlab.normpdf(x1, mu_real, sigma_real))
        plt.plot(x2,mlab.normpdf(x2, mu_fake, sigma_fake))
        plt.title('Generate 1D Gaussian Distribution using GAN: %7d epoch'%(epoch+1))
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        plt.savefig('img/Generate 1D Gaussian Distribution using GAN: %7d epoch'%(epoch+1) + '.png',dpi=200)
        plt.show()
~~~

---
## Visualization

<iframe src="//www.slideshare.net/slideshow/embed_code/key/x7VFNpDIQ8mc2D" width="340" height="290" frameborder="0" marginwidth="0" marginheight="0" scrolling="no" style="border:1px solid #CCC; border-width:1px; margin-bottom:5px; max-width: 100%;" allowfullscreen> </iframe> <div style="margin-bottom:5px"> <strong> <a href="//www.slideshare.net/SungnamPark2/gan-1d-gaussian-distribution-tutorial" title="[GAN] 1D Gaussian Distribution Tutorial" target="_blank">[GAN] 1D Gaussian Distribution Tutorial</a> </strong> from <strong><a target="_blank" href="//www.slideshare.net/SungnamPark2">Sungnam Park</a></strong> </div>

파란색 선이 실제 가우시안 분포이고, 노란색 선이 우리가 점점 학습해서 만들어내가는 가짜 가우시안 분포입니다. 처음에는 어처구니 없는 값을 보이다가 점점 비슷해지는 것을 보실 수 있습니다. 만약 GAN을 더 배우신다고 해도, 기반이 되는 원리는 이와 비슷합니다. 학습하는 레이러가 좀더 복잡해지고, 때로는 Generator와 Discriminator가 많아지기도 하고, 최적화하는 값이 달라지기도 하고, noise를 조정하기도 합니다. 좀 어렵지만, 그래도 GAN이 어떤 느낌적인 느낌인지를 알 수 있는 시간이었길 바랍니당.

---
## Reference
[GAN keras code]
 [https://hussamhamdan.wordpress.com/2017/04/29/generative-adversarial-networks-keras-code/](https://hussamhamdan.wordpress.com/2017/04/29/generative-adversarial-networks-keras-code/)

[GAN tf code]
[https://github.com/hwalsuklee/tensorflow-GAN-1d-gaussian-ex](https://github.com/hwalsuklee/tensorflow-GAN-1d-gaussian-ex)

[GAN pytorch code]
 [https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f](https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f)

---
