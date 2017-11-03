---
layout: post
title: GAN colorization
excerpt: "GAN으로 흑백이미지에 색을 입혀봤습니다."
categories: [GAN]
comments: true
use_math: true
---

# GAN으로 흑백 이미지 색칠하기

<h6 align="right">강병규</h6>

안녕하세요. 한달 전에 [GAN으로 흑백 이미지 색칠하기](https://github.com/YBIGTA/Deep_learning/blob/master/GAN/2017-09-16-gan-colorization.md)라는 글을 올렸었습니다. 그때 결과물은 아직 학습이 진행이 안돼 나중에 추가하겠다고 했는데요, 들어가보신 분이 계실진 모르겠지만 아직까지 결과물을 올리지 못했습니다. 그 이유는

![fake_samples_epoch_69 900](https://user-images.githubusercontent.com/25279765/32103098-f180bc3c-bb59-11e7-936a-a8eeea2bfea8.png)

> 69번째 에폭에서의 결과물

결과물이 제대로 안나왔기 때문이었습니다. 대충 윤곽은 따지는 것 같은 느낌에 희망을 갖고 좀 더 돌려봤지만 갱생의 여지가 없는 코드로 판명됐습니다. 그래서 오늘은 **왜 실패했는가?** 부터 시작해보려고 합니다.

## 왜 실패했는가?

우선 첫번째로 참고한 [코드](https://github.com/aleju/colorizer)가 lua기반의 torch로 짜여진 코드였기 때문입니다. 기본적인 네트워크 구조는 pytorch로도 쉽게 따라할 수 있지만, 학습과정에서 세부적인 부분들은 제가 lua를 모르기 때문에 놓친 내용이 있을지도 모릅니다.

두번째로는 결과물을 **RGB** 로 만들었기 때문입니다. "이미지면 당연히 RGB아닌가?"라고 생각하실 수도 있는데요, 아닙니다. 정말로 많은 표현법이 존재하고요 이를테면 HSV나 YUV같은 방법들로도 이미지를 표현할 수가 있습니다. 여기서 RGB를 사용한 것이 왜 문제가 되냐면 만들어내야하는게 RGB 세 차원 전부가 되기 때문이죠. 즉 우리가 가진 흑백 이미지 정보를 활용할 수 없습니다. 이 얘기는 밑에서 조금 더 자세히 해보기로 합시다.

세번째로는 **Loss** 를 제대로 정의하지 못했습니다. 물론 GAN에서 기본적인 Loss는 어디든 동일합니다만 이 경우에는 L1 regularization을 추가적으로 해줘야했습니다.간단히 얘기해보겠습니다. 위에서 말했듯, 우리는 흑백 이미지를  **바탕** 으로 이미지를 색칠합니다. 하지만 이 과정에서 색칠한 이미지가 원래의 흑백이미지를 제대로 반영하고 있다는 보장이 없습니다. 따라서 L1 regularization으로 색칠한 결과물과 원래의 흑백 이미지 간의 차이를 추가해줘야했습니다. 이번에 다시 짜면서 제대로 보니 위에서 참고한 코드에서 L1 regularization을 해주더라구요..

![image](https://user-images.githubusercontent.com/25279765/32105811-b614d682-bb64-11e7-82f5-21e40572c24d.png)

실제로 pix2pix 등의 논문을 봅시다. 구글 지도를 바탕으로 항공사진을 만든다고 했을 때 우리가 만든 항공 사진이 원래의 구글 지도와 얼마나 똑같은지를 반영해줄 필요가 있습니다. 따라서 실제로 loss를 정의할때 L1 regularization을 더해줍니다.

이번에 제 차례가 돌아오면서 뭘 할지 고민을 좀 했습니다. 색칠에 성공하지 못한 것이 아쉬웠는데 제 눈에 띈 논문이 하나 있었습니다.

![image](https://user-images.githubusercontent.com/25279765/32103557-e26d97f4-bb5b-11e7-9628-e1cc9da6cd52.png)

올해 2월에 나온 [논문](https://arxiv.org/pdf/1702.06674.pdf)입니다. 이름부터 뭘할지 느낌이 팍 오는데 왜 저번 구현에서는 못봤는지 의문입니다. 여튼 오늘은 이 논문을 간단하게 리뷰해보고 논문에서 제시된 구조를 바탕으로 학습시킨 결과물을 진짜로 보여드리겠습니다. 이번에는 진짜입니다. 진짜로요.

## Unsupervised Diverse Colorization via Generative Adversarial Networks

이 논문에 따르면 우리의 task부터 다시 정의할 필요가 있습니다. 우리가 활용하는 흑백이미지는 컬러 이미지를 흑백처리한 것이었습니다. 여기서 흑백이미지를 색칠하는 것인데요, 생각해보면 우리는 색칠할때 "원래" 이미지로 돌아가기를 원합니다. 최소한 원래 이미지와 유사하기를 원하죠. 하지만 그건 불가능하다고 주장합니다. 이미지를 흑백처리하는 공식을 다시 생각해봅시다.

$${ G = 0.2989 * R + 0.5870 * G  + 0.1140 * B} $$

자, RGB값으로 G를 만들 수 있는 건 분명합니다. 이때 우리가 G값을 알면 RGB값을 알 수 있을까요? 정답은 당연히 "아니요"입니다. 같은 grayscale값을 갖는 RGB 경우의 수는 정말 많습니다. 즉 우리가 어떤 흑백이미지를 보고 "얘는 원래 빨강이네, 쟤는 원래 초록이네"할 수 없다는 의미입니다.

![image](https://user-images.githubusercontent.com/25279765/32104409-45d7224e-bb5f-11e7-9d42-72e62c6db7f9.png)
> 퀴즈: 가운데 남자가 입은 양복 색깔은?

그래서 이 논문은 목표를 "원래의 이미지로 복구하기"가 아닌, "진짜같이 다양한 색깔을 사용하기"로 재정의합니다(As our goal alters from producing the original colors to producing realistic diverse colors)

![image](https://user-images.githubusercontent.com/25279765/32104695-866ee67e-bb60-11e7-8f9c-f3fddc1cae7a.png)

기본적인 매커니즘은 저번과 동일합니다. Conditional GAN에서 흑백이미지를 조건으로 주고, 노이즈와 함께 Generator에 입력으로 넣습니다. Generator는 이에 따라 이미지를 색칠할 것이고, Discriminator는 진짜 이미지와 가짜 이미지를 보고 판단을 합니다.

하지만 여기서 이 논문은 Generator의 효율성을 위해 RGB대신 YUV를 사용합니다.

### YUV?

YUV란게 아마 다들 생소하실 것 같은데요, 이 또한 색을 표현하는 방식입니다. 여기서 중요한점은 Y가 **흑백** 그 자체라는 점입니다. 즉 Y라는 흑백 이미지 위에 U, V 필터를 얹어 컬러 이미지를 표현한다고 생각하면 됩니다. 즉, 우리가 만약 이미지를 YUV로 처리한다면 번거롭게 이미지를 흑백처리하는 과정을 생략하고 Y만 넣어주면 된단 얘기가 되는거죠.

장점은 이것뿐만이 아닙니다. 우리는 흑백 이미지를 색칠하는 건데 YUV를 사용하게 되면 이미 Y는 주어진 것입니다. 즉 RGB에서는 세 channel모두 예측해야 했지만 여기서는 U, V 2개의 channel만을 만들어내면 되는 것이죠.

만약 RGB를 쓰고 싶다면은 위에서 언급했듯이 L1 regularization을 해줘야합니다.
### Detail

![image](https://user-images.githubusercontent.com/25279765/32105402-485c1552-bb63-11e7-9a63-d76c1fc2498e.png)

저번 구현에서 사용했던 구조는 Encoder-Decoder와 유사한 느낌이었습니다. Deconvolution을 사용하지는 않았지만요. 여기서는 그냥 일관성있게 조건, 즉 흑백 이미지를 모든 레이어에 계속 넣어줍니다. 또 Generator는 Convolution layer를 사용하는데, 이때 각 레이어에 입력으로 들어가는 Tensor들은 모두 동일한 가로 세로 크기를 갖는다는 점입니다. 논문에서는 흑백 이미지에서 어떤 중요한 spatial 정보를 그대로 전달해주기 위함이라고 이야기합니다.

![image](https://user-images.githubusercontent.com/25279765/32105887-0ee325ac-bb65-11e7-8341-07e1f0deebe4.png)
> 큐브모양은 Convolution-BatchNorm-ReLU를 의미합니다

구체적인 구조는 이렇게 생겼습니다. 뭐 딱히 어려운 부분은 없습니다. 주목할 만한 부분은 noise를 처음에만 넣어주는게 아니라 한번 더 넣어준다는 것인데요, 논문에 따르면 noise를 처음에만 넣어주게 될 경우 네트워크를 거치면서 noise를 잊게 될 수 있다고 얘기를 합니다. 이를 해결하기 위해 Multilayer noise를 도입한 것이죠. 또 Multilayer condition을 넣어줘 흑백 이미지에 대한 정보를 지속적으로 파악할 수 있게 해줍니다.

Generator를 보면 100개의 노이즈를 뽑아서 얘네를 Fully-connected에 넣고 64 x 64로 만들어줍니다. 이렇게 만든 노이즈를 흑백 이미지와 합쳐서 CNN에 넣고, 넣고, 넣고... 마지막에 2개의 채널로 만들어 U와 V 역할을 하게 만들어주면 Generator의 역할은 끝입니다. Discriminator는 더 간단합니다. 그냥 이미지를 넣고 CNN을 통해 하나의 확률값을 만들어내게 하죠.

특별히 이야기할만한 부분들은 다 했습니다. 이제 본격적인 구현으로 넘어가보죠.

### Implementation using Pytorch

![yuv](https://wikimedia.org/api/rest_v1/media/math/render/svg/66a4f0f56ff9cc473a473965c4cbf32d4b4345ca)

Pytorch를 사용했습니다. 데이터셋으로는 STL10을 사용했습니다. 그냥 일반적인 object들 사진이 들어있다고 생각하시면 됩니다.근데 코드를 짜는 과정에서 예상치 못한 문제가 생겼습니다. 원래 이미지는 RGB인데 얘네를 YUV로 바꿔줄 방법이 없어요. 위키피디아에 RGB를 YUV로 처리하는 matrix가 있긴 해서 이들을 numpy 연산으로 구현했습니다. 이렇게 해서 문제가 해결되었다면 얼마나 좋을까요... 밑에 사진을 보시죠.

![original](https://user-images.githubusercontent.com/25279765/32106895-f7a88406-bb67-11e7-81d4-5cd7e508ef21.jpg)
![rgb2yuv2rgb](https://user-images.githubusercontent.com/25279765/32106893-f6abfc36-bb67-11e7-888d-3d52732585d6.jpg)
> 상 - 원래 RGB이미지, 하 - RGB이미지를 YUV로 바꾸고 이를 다시 RGB로 바꾼 이미지

보면 색깔이 B612필터를 끼얹은 마냥 레트로 느낌이 나게 변해버렸습니다. numpy연산과정에서 정보의 손실이 발생한건지 식이 잘못된건지 뭐가 문젠지 모릅니다. 그러니까 우리도 논문처럼 task를 redefine합시다. "GAN으로 흑백이미지 (레트로 느낌으로) 색칠하기"로 말이죠.

코드로 넘어갑시다.

```python
import pickle
from __future__ import print_function
import itertools
import math

import torch
from torch import optim
import torchvision
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.autograd import Variable

import os
import numpy as np
```
> 필요한 패키지들부터 가져옵시다.

```python
def toYUV(rgb):
    rgb = rgb.numpy()
    R, G, B = rgb[0, :, :], rgb[1, :, :], rgb[2, :, :]
    Y = 0.299 * R + 0.587 * G + 0.114 *B
    U = -0.147 * R + -0.289 * G + 0.436 * G
    V = 0.615 * R + -0.515 * G - 0.100 * B
    return torch.from_numpy(np.asarray([Y, U, V]).reshape(3, 64, 64))
```
```python
def toRGB(yuv, batchsize):
    """shape of yuv is bs x 3 x 64 x 64, ordered by YUV"""
    lst = []
    for data in yuv:
        Y, U, V = data[0, :, :], data[1, :, :], data[2, :, :]
        R = Y + 1.140 * V
        G = Y + (-0.395 * U) + (-0.581 * V)
        B = Y + 2.032 * U
        lst.append([R,G,B])
    return np.asarray(lst).reshape(batchsize, 3, 64, 64)#.clip(0, 255)
```
> 위는 RGB를 YUV로 바꾸는 코드, 밑은 YUV를 RGB로 바꾸는 코드

```python
transform = transforms.Compose([
    transforms.Scale(64),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: toYUV(x))
])
train_dataset = dsets.STL10('./data/', split="train+unlabeled", transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,shuffle=True)
```
> YUV로 바꿔주는 과정은 pytorch에 없으므로 위에서 만든 함수를 사용합시다.

```python
def extractGray(batchSize, yuv):
    lst = []
    for data in yuv:
        lst.append(data[0])
    return np.asarray(lst).reshape(batchSize, 1, 64, 64)
```
> YUV에서 Y만 뽑아냅니다.

```python
class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()
        self.cnn = nn.Sequential(
            # 3 x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64 x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 128 x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 256 x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)

            # 512 x 4 x 4
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1),
            nn.Sigmoid()
        )
    def forward(self, input):
        # input is real or fake colored image
        x = self.cnn(input)
        x = x.view(x.size(0), 512 * 4 * 4) # flatten it
        output = self.fc(x)
        return output.view(-1,1).squeeze(1)
```
> Discriminator - 위의 사진을 보면 마지막에 FC를 사용한다는데 몇층을 쌓았다는 언급이 없어 그냥 한 층으로 했습니다. 마지막에는 확률값이 나와야하니 시그모이드를 사용했구요

```python
class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()

        self.fc = nn.Linear(100, 1 * 64 * 64)
        self.conv1 = nn.Conv2d(2, 130, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(130)

        self.conv2 = nn.Conv2d(132, 66, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(66)

        self.conv3 = nn.Conv2d(68, 65, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(65)

        self.conv4 = nn.Conv2d(66, 65, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(65)

        self.conv5 = nn.Conv2d(66, 33, 3, 1, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(33)

        self.conv6 = nn.Conv2d(34, 2, 3, 1, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, input, noise_pure):
        # input is grayscale image(Y of YUV), noise is random sampled noise
        noise = self.fc(noise_pure)
        noise = noise.view(noise.size(0), 1, 64, 64)

        # 2 x 64 x 64
        x = self.conv1(torch.cat([input, noise], dim=1))
        x = self.bn1(x)
        x = self.relu(x)

        # 130 x 64 x 64
        input2 = torch.cat([input, x ,noise], dim=1)
        # 132 x 64 x 64
        x = self.conv2(input2)
        x = self.bn2(x)
        x = self.relu(x)

        # 66 x 64 x 64
        input3 = torch.cat([input, x, noise], dim=1)
        # 68 x 64 x 64
        x = self.conv3(input3)
        x = self.bn3(x)
        x = self.relu(x)

        # 65 x 64 x 64
        input4 = torch.cat([input, x], dim=1)
        # 66 x 64 x 64
        x = self.conv4(input4)
        x = self.bn4(x)
        x = self.relu(x)

        # 65 x 64 x 64
        input5 = torch.cat([input, x], dim=1)
        # 66 x 64 x 64
        x = self.conv5(input5)
        x = self.bn5(x)
        x = self.relu(x)

        # 33 x 64 x 64
        input6 = torch.cat([input, x], dim=1)
        # 34 x 64 x 64
        x = self.conv6(input6)

        output = torch.cat([input, x], dim=1)
        return output
```
> Generator, 매 레이어마다 흑백이미지를 계속 넣어줍니다. noise는 처음의 100개의 값이고 FC를 거쳐 64x64가 됩니다.

```python
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:         # Conv weight init
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:  # BatchNorm weight init
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
netG = _netG().cuda()
netG.apply(weights_init)
print(netG)

netD = _netD().cuda()
netD.apply(weights_init)
print(netD)
```
> weight를 초기화하고 G와 D를 만들어줍시다.

```python
criterion = nn.BCELoss().cuda()

input = torch.FloatTensor(batchSize, 3, 64, 64).cuda()
noise = torch.FloatTensor(batchSize, 100).cuda()

label = torch.FloatTensor(batchSize).cuda()
real_label = 1
fake_label = 0
optimizerD = optim.Adam(netD.parameters(), lr=0.0002,betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002,betas=(0.5, 0.999))
result_dict= {}
loss_D, loss_G = [], []
```
> loss를 계산하는 기준을 선언하고 optimizer도 만들어줍시다.

```python
for epoch in range(1,300):
    for i, (data, _) in enumerate(train_loader):
        data = data.cuda()
        batchSize = len(data)
        gray = extractGray(batchSize, data.cpu().numpy())
        grayv = Variable(torch.from_numpy(gray)).cuda()
        #############
        # D!        #
        #############
        netD.zero_grad()
        ##############
        # real image #
        ##############
        input.resize_as_(data).copy_(data)
        label.resize_(len(data)).fill_(real_label)

        inputv = Variable(input).cuda()
        labelv = Variable(label).cuda()

        output = netD(inputv)
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        ##############
        # fake image #
        ##############
        noise.resize_(batchSize, 100).uniform_(0,1)
        noisev = Variable(noise).cuda()

        # create fake images
        fake = netG(grayv, noisev)

        # cal loss
        output = netD(fake.detach())
        labelv = Variable(label.fill_(fake_label)).cuda()
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()

        errD = errD_real + errD_fake
        optimizerD.step()

        ##############
        # G!         #
        ##############
        netG.zero_grad()
        labelv = Variable(label.fill_(real_label)).cuda()
        output = netD(fake)

        errG = criterion(output, labelv)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()

        if ((i+1) % 100 == 0):
            if not os.path.exists('results/'):
                os.makedirs('results/')
            rgb = toRGB(fake.cpu().data.numpy(), batchSize)
            vutils.save_image(torch.from_numpy(rgb), '%s/fake_samples_epoch_%s.png' % (outf, str(epoch)+" "+str(i+1)))
    print(epoch)
    print(errD.data[0], errG.data[0])
    rgb = toRGB(fake.cpu().data.numpy(), batchSize)
    vutils.save_image(torch.from_numpy(rgb),'%s/fake_samples_epoch_%s.png' % (outf, epoch))
    loss_D.append(errD.data[0])
    loss_G.append(errG.data[0])
    result_dict = {"loss_D":loss_D,"loss_G":loss_G}
    pickle.dump(result_dict,open("./{}/result_dict.p".format(outf),"wb"))
    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG.pth' % (outf))
    torch.save(netD.state_dict(), '%s/netD.pth' % (outf))
```
> 크게 어려운 부분은 없습니다. 흑백 이미지는 YUV에서 Y와 같으므로 배치 전체에서 Y만 뽑아내는 함수를 사용합니다.

### Result

이렇게 학습을 진행한 결과 나온 이미지들은 다음과 같습니다.

![fake_samples_epoch_1](https://user-images.githubusercontent.com/25279765/32107788-9b45c00e-bb6a-11e7-96dc-5dae6f484a65.png)
![fake_samples_epoch_23 100](https://user-images.githubusercontent.com/25279765/32107791-a0274192-bb6a-11e7-9de1-cef699721075.png)
![fake_samples_epoch_73](https://user-images.githubusercontent.com/25279765/32107794-a2c47b0e-bb6a-11e7-996b-01437b16df90.png)
![fake_samples_epoch_80 900](https://user-images.githubusercontent.com/25279765/32107800-a58887c2-bb6a-11e7-9fbe-b88089a2ccef.png)
> 순서대로 1, 23, 73, 80번째 에폭

학습 초기 과정에서는 하양과 검정을 먼저 인식하는 듯한 모습을 보여주더니 뒤로 갈수록 나름 진짜같은 색감이 나옵니다. 위에서 보셨듯이 YUV와 RGB간의 Conversion이 완벽하지 않으니 결과를 볼 때도 이를 좀 감안해야합니다. 어쨌든 위에서처럼 레트로 색감으로 이미지가 칠해졌습니다. 아니라고요? 칠해졌다고 칩시다.

### Conclusion

이렇게 해서 색칠하기는 성공했습니다. 아쉬운 점이 있다면 YUV와 RGB간의 Conversion에서 뭔가가 잘못됐다는 점? 다시 구현해볼 시간이 있다면 다음에는 RGB를 그대로 사용하고 L1 regularization을 걸어보도록 하겠습니다. 감사합니다!
