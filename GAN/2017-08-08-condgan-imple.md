---
layout: post
title: GAN으로 내 핸드폰 번호 손글씨 만들기(feat. pytorch, MNIST)
excerpt: "condtional GAN과 MNIST데이터를 pytorch를 이용해 구현해보았습니다."
categories: [GAN]
comments: true
use_math: true
---
# GAN으로 내 핸드폰 번호 손글씨 만들기 (feat. pytorch)
<h6 align="right">강병규</h6>

이번에는 GAN과 MNIST 데이터를 이용해서 손글씨 숫자를 학습을 시키고, 핸드폰 번호를 만들어 보도록 하겠습니다. pytorch를 사용할 거구요. 그전에 잠깐 되짚어 볼 것이 있습니다.

## 기존 GAN의 문제점

기존 GAN에 대해서는 [이전 글](https://kangbk0120.github.io/articles/2017-07/first-gan)을 참조하세요.
기존 GAN의 문제점은 내가 만들고 싶은 데이터를 만들어내지 못한다는 것에 있습니다. 예를 들어 MNIST 데이터로 학습시킨 GAN이 있다고 해봅시다. 이때 GAN을 이용해서 내가 만들고 싶은 숫자를 만들어내지 못합니다. 결국 내가 원하는 숫자가 나올 때까지 입력이 되는 noise를 계속해서 바꿔줘야만 하는 것입니다.

이러한 문제를 해결하기 위해 [Conditional GAN](https://arxiv.org/abs/1411.1784)이라는 모델이 등장했습니다.  단순히 어떤 추가적인 정보를 넣어주기만 하면 내가 원하는 데이터를 만들어줄 수 있는 것이죠.

사진을 보면 더 명확하게 이해할 수 있습니다.
![cond-gan](https://user-images.githubusercontent.com/25279765/28810813-acb1b642-76c6-11e7-94cc-cf5cdeb579d4.PNG)
그냥 Discriminator와 Generator에 어떠한 정보 y만 넣어주면 내가 원하는 데이터를 만들어 낼 수 있습니다. 이때 주목해야할 점은 기존의 GAN과 목적함수가 달라진다는 것 입니다. y라는 추가적인 정보가 들어갔으므로, 조건부확률이 된다는 것만 주의하시면 됩니다. 더 알고 싶으시다면 [이전 글](https://kangbk0120.github.io/articles/2017-08/conditional-gan)을 참조하세요

## 구체적인 Implementation

이제 본격적으로 시작해봅시다. 우선 필요한 모듈들을 가져와야합니다.
```python
import itertools
import math
import time

import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from IPython import display
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os

%matplotlib inline
```

우리는 pytorch를 사용할 것이므로 pytorch를 가져옵니다. 이때 왜 tensorflow의 mnist를 가져오는지 궁금해하실 수도 있을 것 같은데요, y를 사용하기 위해서는 label들을 one-hot vector로 만들어주는 것이 좋습니다. 손글씨 숫자는 0~9의 label을 가지겠죠? 만약 어떤 데이터가 3이었다면, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]의 값을 가지게 되는 것입니다. pytorch의 dataset에도 MNIST가 있지만, 이를 one-hot vector로 만들어주는 과정이 조금 지저분해서 tensorflow의 dataset을 이용하고자 합니다.

```python
mnist = input_data.read_data_sets('../../data', one_hot=True)
```
이렇게 해주면 mnist라는 변수안에 MNIST데이터를 저장할 수 있습니다.
```python
print(mnist.train.images.shape)
print(mnist.train.labels.shape)
```
를 해주면 총 55000개의 데이터가 있고, 이미지는 흑백의 28 x 28이므로 784, 그리고 label은 0~9이므로 10의 값을 가진다는 것을 볼 수 있습니다.

그 다음으로 Discriminator를 정의하겠습니다. 위에서 언급했던 논문의 구조를 그대로 구현해보려했지만 구체적으로 제시되어있지않아 제 임의로 설계했습니다.
```python
class Discriminator(nn.Module):
    """Discriminator, 논문에 따르면 maxout을 사용하지만
       여기서는 그냥 Fully-connected와 LeakyReLU를 사용하겠습니다.
       논문에서는 Discriminator의 구조는 그렇게 중요하지 않다고 말합니다"""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784+10, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x, y):
        x = x.view(x.size(0), 784)
        out = self.model(torch.cat([x,y], 1))
        out = out.view(out.size(0), -1)
        return out
```

이때 첫번째 layer를 보면 784+10차원의 값을 받는다는 것을 볼 수 있습니다. 위에서 보셨듯 이미지 하나는 784의 값을 가지고 label은 10의 값을 가지므로, 이들을 함께 넣어주기 위해서 784+10의 값을 가지는 것입니다. 마지막 층 전에는 활성화 함수로 LeakyReLU를 사용하며 Dropout을 사용했습니다.

Discriminator는 어떠한 데이터가 진짜인지 가짜인지 판단해야하므로 한 개의 확률값을 만들어내야만 합니다. 따라서 마지막에는 1개의 값으로 만들어주고 이를 0~1사이의 확률값으로 만들어주기 위해 Sigmoid를 사용했습니다.

이제 Generator를 정의할 차례입니다.
```python
class Generator(nn.Module):
    """Generator, 논문에 따르면 100개의 noise를 hypercube에서 동일한 확률값으로 뽑고
       z를 200개, y를 1000개의 뉴런으로 전달합니다. 이후 1200차원의 ReLU layer로 결합하고
       Sigmoid를 통해 숫자를 만들어냅니다."""
    def __init__(self):
        super().__init__()
        self.map = nn.Sequential(
            nn.Linear(100+10, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, 512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1024, 784),
            nn.Tanh()
        )
    def forward(self, z, y):
        out = self.map(torch.cat([z, y], 1))
        return out
```
noise 값은 100고 원하는 데이터를 만들어줘야하므로 label을 포함해야합니다. 따라서 100+10차원의 값들을 첫번째 layer에서 처리합니다. 이미지는 784개의 값들을 가지므로, 마지막 layer에서는 784개의 값들을 만들어내도록 합니다. 여기서도 Discriminator처럼 Dropout을 적용했습니다.

```python
discriminator = Discriminator()
generator = Generator()

criterion = nn.BCELoss()      
lr = 0.0002
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
```
그리고 모델과 loss, 최적화방법을 선언합시다. 0이면 가짜, 1이면 진짜를 구분하는 것이므로 loss는 BCELoss로 했습니다. 또한 Optimizer는 Adam을 사용했습니다.

```python
def train_discriminator(discriminator, x, real_labels, fake_images, fake_labels, y):
    discriminator.zero_grad()
    outputs = discriminator(x, y)
    real_loss = criterion(outputs, real_labels)
    real_score = outputs

    outputs = discriminator(fake_images, y)
    fake_loss = criterion(outputs, fake_labels)
    fake_score = fake_loss

    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss, real_score, fake_score

def train_generator(generator, discriminator_outputs, real_labels, y):
    generator.zero_grad()
    g_loss = criterion(discriminator_outputs, real_labels)

    g_loss.backward()
    g_optimizer.step()
    return g_loss
```

Discriminator와 Generator를 학습시키는 방법을 정의합니다. Discriminator는 진짜 데이터를 보고 이것이 진짜인지 가짜인지 판단을 하고 거기서 loss가 발생합니다. 그 다음 가짜 데이터를 보고 이것이 진짜인지 가짜인지를 판단하고 이과정에서 또한 loss가 발생합니다. 이후 이 두 loss를 합한다음 역전파시키고 parameter들을 업데이트합니다.

Generator는 가짜데이터를 만들어낸다음, Discriminator에 전달해 Discriminator의 판단을 기준으로 loss를 구하고 역전파, 업데이트합니다.

```python
# 결과를 jupyter notebook에 띄우기 위한 코드입니다.
num_test_samples = 9
size_figure_grid = int(math.sqrt(num_test_samples))
fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(6, 6))
for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
    ax[i,j].get_xaxis().set_visible(False)
    ax[i,j].get_yaxis().set_visible(False)


num_epoch = 200
batch_num = 5500
num_fig = 0

tracking_dict = {}
tracking_dict["d_loss"] = []
tracking_dict["g_loss"] = []
tracking_dict["real_score"] = []
tracking_dict["fake_score"] = []

for it in range(120000):
    z = Variable(torch.randn(100, 100))
    X, y = mnist.train.next_batch(100)
    X = Variable(torch.from_numpy(X).float())
    y = Variable(torch.from_numpy(y).float())

    fake_images = generator(z, y)

    real_labels = Variable(torch.ones(100))
    fake_labels = Variable(torch.zeros(100))

    # Discriminator 학습
    d_loss, real_score, fake_score = train_discriminator(discriminator, X, real_labels,\
                                                         fake_images, fake_labels,y)

    z = Variable(torch.randn(100, 100))
    fake_images = generator(z, y)
    outputs = discriminator(fake_images, y)

    g_loss = train_generator(generator, outputs, real_labels, y)
    # 100번마다 결과를 출력합니다
    # 임의로 9개의 noise를 뽑고,
    # 1~9까지를 one-hot encoding한다음 concat합니다
    if (it+1) % 100 == 0:
        z = Variable(torch.randn(9, 100))
        c = np.eye(9, dtype='float32')
        c = np.c_[np.zeros(9), c]
        test_images = generator(z, Variable(torch.from_numpy(c).float()))

        # 이미지를 쥬피터 노트북에 띄웁니다.
        if not os.path.exists('results/'):
            os.makedirs('results/')
        for k in range(num_test_samples):
            i = k//3
            j = k%3
            ax[i,j].cla()
            ax[i,j].imshow(test_images[k,:].data.cpu().numpy().reshape(28, 28), cmap='Greys')
        display.clear_output(wait=True)
        display.display(plt.gcf())

        plt.savefig('results/mnist-gan-%03d.png'%num_fig)
        num_fig += 1
        print('step: %d d_loss: %.4f, g_loss: %.4f, '
              'D(x): %.2f, D(G(z)): %.2f'
              %(it+1, d_loss.data[0], g_loss.data[0],
                real_score.data.mean(), fake_score.data.mean()))
        tracking_dict["d_loss"].append(d_loss.data[0])
        tracking_dict["g_loss"].append(g_loss.data[0])
        tracking_dict["real_score"].append(real_score.data.mean())
        tracking_dict["fake_score"].append(fake_score.data.mean())
```

학습시키는 과정에서 나온 결과물들은 다음과 같습니다
<br>
![mnist-gan-1062](https://user-images.githubusercontent.com/25279765/29065720-eaeb0fca-7c67-11e7-953e-c491e00a619b.png)
<br>
![mnist-gan-1082](https://user-images.githubusercontent.com/25279765/29065722-ede5560e-7c67-11e7-9537-e2d06ac47ea7.png)
<br>
이쁘진 않지만 그래도 1~9라는 숫자를 알아볼 수 있을 정도로는 결과물들이 나왔음을 알 수가 있습니다.

아까 Generator를 정의할 때 Dropout을 설정했으므로
```python
Generator.eval()
```
이렇게 하면 Generator에서 Dropout layer를 끌 수 있습니다. 이렇게해서 얻을 수 있는 최종 결과물은
<br>
![mnist-gan-finalresult](https://user-images.githubusercontent.com/25279765/29065725-f07ab8aa-7c67-11e7-8893-29ea930ffb07.png)
<br>
이었습니다.

이제 마지막과정입니다. 이번 글의 목표는 내 핸드폰 번호를 MNIST를 통해 만들어내는 것이었으므로 이제 condtion에 내 번호를 넣어주고, 이를 noise와 함께 넣어주기만 하면 됩니다.

```python
c = np.zeros([8,10]) # 010은 빼고 8자리만 하겠습니다.
c[0, 5] = 1
c[1, 7] = 1
c[2, 0] = 1
c[3, 1] = 1
c[4, 3] = 1
c[5, 9] = 1
c[6, 8] = 1
c[7, 6] = 1
# 뒷부분에 숫자를 넣어주기만 하면 됩니다.
```

```python
generator.train()
```
을 해주면 다시 Dropout을 적용할 수 있습니다. 이렇게하면 실행시킬 때마다 다른 결과를 얻을 수 있습니다.
```python
num_test_samples = 16
size_figure_grid = int(math.sqrt(num_test_samples))
fig, ax = plt.subplots(2, 4, figsize=(6, 6))
for i, j in itertools.product(range(2), range(4)):
    ax[i,j].get_xaxis().set_visible(False)
    ax[i,j].get_yaxis().set_visible(False)
z = Variable(torch.randn(8, 100))
test_images = generator(z, Variable(torch.from_numpy(c).float()))
for k in range(num_test_samples):
    i = k//4
    j = k%4
    try:
        ax[i,j].cla()
        ax[i,j].imshow(test_images[k,:].data.cpu().numpy().reshape(28, 28), cmap='Greys')
    except:
        pass
display.clear_output(wait=True)
display.display(plt.gcf())
plt.savefig('results/mnist-gan-phone.png')
```
이렇게 해주면 최종 결과를 얻을 수 있습니다.

![mnist-gan-phone 2](https://user-images.githubusercontent.com/25279765/29065968-cb989ac4-7c68-11e7-8fd2-0402e09a53ed.png)

## 정리

오늘은 conditional-gan을 구현해봤습니다. 깔끔한 output을 만들어내는데에는 실패했지만 어느 정도의 결과물은 얻을 수 있었습니다.
