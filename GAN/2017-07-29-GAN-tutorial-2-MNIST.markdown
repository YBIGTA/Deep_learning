# GAN으로 MNIST 이미지 생성하기
작성자: YBIGTA 10기 김지중



<span style="color:#7C7877; font-family: 'Apple SD Gothic Neo'; font-weight:200">

<br>

지난 시간 1차원의 정규분포를 생성하는 GAN의 TOY MODEL에 대해 살펴보았습니다. 오늘은 이미지 데이터를 생성해볼텐데요, MNIST 데이터 셋을 이용하여 숫자 필기체 이미지를 생성할 겁니다. 지난번과 마찬가지로 Pytorch를 이용했습니다.


#### Specification
MNIST 정도의 데이터는 로컬에서도 돌아갑니다. 시간이 많이 걸릴 따름입니다. 제 맥북 에어에서는 5시간 정도 걸렸는데, 마지막 코드 한 줄에서 커널이 죽고 말았습니다. 때문에 loss 및 accuracy 정보가 날라갔는데, 도저히 로컬에서 다시 돌릴 엄두가 나지 않아 AWS EC2 Instance에서 작업했습니다. 인스턴스는 *p2.xlarge*, AMI는 *Deep Learning AMI Ubuntu Version* 을 골랐습니다. 요 Image에 Pytorch는 설치되어있지 않아서, Pytorch는 새로 설치하였습니다. 작업 환경의 스펙은 아래와 같습니다.

- **GPU** NVIDIA K80
- **Ubuntu** 16.04
- **CUDA** 8.0
- **Pytorch** 0.1.12

코드의 기본적인 WorkFlow는 아래와 같습니다.
<br>
<br>


## Workflow
1. 실제 이미지들과 fake 이미지들을 샘플링합니다.<br><br>
    * 실제 이미지는 데이터셋에서 Load합니다.
    * fake 이미지는 Generator에 noise라는 인풋을 넣어서 만듭니다.
<br><br>

2. Discriminator를 학습시킵니다.<br><br>
    (1) 실제 이미지들을 넣고 분류기를 돌려봅니다.
      * real_loss: 실제 이미지들을 넣은 결과값들(0 혹은 1로 구성된 벡터)와 실제 이미지들의 레이블(1로 이루어진 벡터)를 비교해서 계산된 loss<br><br>

    (2) fake 이미지들을 넣고 분류기를 돌려봅니다.
      * fake_loss: fake 이미지들을 넣은 결과값(0 혹은 1로 구성된 벡터)와 fake 이미지들의 레이블(영벡터)를 비교해서 계산된 loss<br><br>

    (3) Discriminator's loss =  real_loss + fake_loss<br><br>

    (4) 오차 역전파 및 파라미터 업데이트
    <br><br>
3. Generator를 학습시킵니다.<br><br>

    (1) 새로운 fake 이미지들을 뽑아서 Discriminator에 일종의 테스트 셋으로 넣어봅니다.
      * fake 이미지는 역시 Generator에 noise를 넣어서 만듭니다.<br><br>

    (2) 테스트 결과값과 실제 이미지의 레이블을 비교해 loss를 계산합니다.<br><br>

    (3) 오차 역전파 및 파라미터 업데이트<br><br><br>

4. Generator가 만든 fake 이미지를 저장합니다.

<br>
<br>

## Modeling

#### 필요한 모듈 불러오기
~~~python
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

%matplotlib inline
~~~

#### 데이터 셋 불러오기
~~~python

transform = transforms.Compose([          
        transforms.ToTensor(),                     
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

train_dataset = dsets.MNIST(root='./data/', train=True, download=True, transform=transform) #전체 training dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True) # dataset 중에서 샘플링하는 data loader입니다
~~~

#### Discriminator 정의
Discriminator는 이미지가 진짜인지 가짜인지 잘 구별하도록 학습됩니다. 보통 MNIST 하면 CNN을 떠올릴 텐데요. 여기서는 non-linearity를 적용한 Fully Connected Layer를 4개 쌓았습니다. MNIST가 흑백의 아주 단순한 이미지이기 때문에  Convolutional Layer를 적용하지 않아도 괜찮은 결과를 보이는 것 같습니다. 지난 "정규분포 근사" 코드에서와는 다르게 Sequential을 활용하여 모델을 설계했습니다. MNIST는 28*28의 이미지입니다. 이를 하나의 벡터로 resizing하면 784차원의 벡터가 될 것입니다. Discriminator는 이 벡터를 input으로 받아 해당 이미지가 실제 이미지일 확률값을 도출합니다.
~~~python
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),                 # MNIST 이미지 사이즈: 28*28 => 일렬로 늘이면 784인 벡터 => 얘를 input으로 받아 1024로 확장
            nn.LeakyReLU(0.2, inplace=True),      # Leaky ReLU => x if x>0 else a*x, 여기서 a는 0.2
            nn.Dropout(0.3),                      # 30% drop out

            nn.Linear(1024, 512),                
            nn.LeakyReLU(0.2, inplace=True),      
            nn.Dropout(0.3),                      

            nn.Linear(512, 256),             
            nn.LeakyReLU(0.2, inplace=True),      
            nn.Dropout(0.3),                      

            nn.Linear(256, 1),                    
            nn.Sigmoid()                          # 시그모이드를 통해 "진짜 이미지일 확률" 추출
        )

    def forward(self, x):
        out = self.model(x.view(x.size(0), 784))  #
        out = out.view(out.size(0), -1)           # 배치사이즈 * 1의 벡터가 결과값으로 나오게 됨
        return out                                
~~~

#### Generator
Generator에도 non-linearity를 적용한 4개의 레이어를 쌓았습니다. size가 100인 noise를 인풋으로 받으면, 레이어를 거쳐 784개의 픽셀값이 나옵니다. 이 픽셀들을 (28,28)로 resizing 해주면 하나의 이미지가 됩니다.
~~~python
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),                  # size가 100인 noise를 인풋으로 받는다, 가중치 벡터를 곱해 256차원 벡터로 확장
            nn.LeakyReLU(0.2, inplace=True),      # Leaky ReLU

            nn.Linear(256, 512),             
            nn.LeakyReLU(0.2, inplace=True),      

            nn.Linear(512, 1024),                 
            nn.LeakyReLU(0.2, inplace=True),      

            nn.Linear(1024, 784),                 
            nn.Tanh()                             
        )

    def forward(self, x):
        x = x.view(x.size(0), 100)                
        out = self.model(x)
        return out
~~~

Discriminator와 Generator를 초기화합니다.
~~~python
discriminator = Discriminator().cuda()                 
generator = Generator().cuda()
~~~

#### Optimizer 설정
1)&nbsp;Loss function<br>
class가 두 개인 classification에서 사용되는 loss function인 BCELoss를 사용하였습니다. 이 모델에서 class는 진짜 / 가짜 두 개 입니다.
<br><br>
2) Optimizer<br>
만능 옵티마이저 Adam을 사용했습니다.<br>

~~~python
criterion = nn.BCELoss()      
lr = 0.0002
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)     
~~~
</br>


#### Loss 계산 및 Training<br>
에 대해 설명하기에 앞서, label에 대해 짚고 넘어갑시다. Discriminator에게 있어서 이 문제는 양자택일의 classification 문제입니다. 진짜, 가짜 두 가지의 class가 존재하는 것입니다. 이 때 진짜를 1, 가짜를 0이라는 label을 달아줍니다. 이 부분에 해당하는 코드는 조금 뒤에 확인할 것입니다. 다시, Discriminator와 Generator를 학습시키는 함수를 살펴봅시다.<br>
<br>
1.&nbsp;Discriminator<br>
앞서 살펴 보았듯, Discriminator에 이미지들의 minibatch를 집어넣으면 해당 이미지가 실제 이미지일 확률이 표현된 확률 벡터가 반환됩니다. 100개 이미지를 넣으면 각각의 이미지가 실제 이미지일 확률이 100개 나온다는 뜻입니다. Discriminator는 이 확률값과 input의 label을 비교하며 loss를 계산합니다.<br>
<br>
    (1) 먼저 실제 이미지들을 Discriminator에 넣어봅니다. 결과값으로 실제 이미지일 확률 벡터가 나옵니다. 이를 모든 element가 실제 이미지의 label(1)인 벡터와 비교하여 BCELoss를 계산합니다.<br>
    <br>
    (2) 다음으로 fake 이미지들을 Discriminator에 넣어봅니다. 결과값으로 실제 이미지일 확률 벡터가 나옵니다. 이를 모든 element가 fake 이미지의 label(0)인 벡터와 비교하여 BCELoss를 계산합니다.<br>
    <br>
    (3) 두 loss를 더해주면 Discriminator의 최종 loss가 됩니다. 이를 역전파하고 parameter를 update 합니다.<br>
<br><br>
2.&nbsp;Generator<br>
Generator의 학습 과정은 아래와 같습니다.<br>
<br>
    (1) 자기가 만든 fake 이미지들을 Discriminator에 넣어봅니다.<br>
    <br>
    (2) 그럼 마찬가지로 Discriminator가 판단하기에 각각의 이미지가 실제 이미지일 확률이 반환됩니다.<br>
    <br>
    (3) Generator의 목적은 Discriminator가 가짜 이미지를 진짜 이미지라고 착각하게 만드는 것입니다. 따라서 아까 계산된 확률값을 실제 이미지 label(1)로 구성된 벡터와 비교하여 BCELoss를 계산합니다. 다시 말하자면, fake 이미지를 집어넣어놓고 마치 이게 실제 이미지였던 것 마냥 loss를 계산하고, 학습하는 것입니다. 이러한 Generator의 loss를 최소화 하는 과정에서 fake 이미지들이 실제 이미지와 유사해집니다.<br>
    <br>

~~~python
def train_discriminator(discriminator, images, real_labels, fake_images, fake_labels):
    discriminator.zero_grad()                     # parameter를 0으로 초기화
    outputs = discriminator(images)               # 실제 image들을 분류기에 넣고 돌린 결과(진짜인지 아닌지) -  1:진짜, 0:가짜
    real_loss = criterion(outputs, real_labels)   # output과 실제 label(1이겠죠)과 비교하여 loss 계산()
    real_score = outputs                          # output의 리스트를 real_score라는 변수에 저장..

    outputs = discriminator(fake_images)          # fake 이미지들을 분류기에 넣고 돌린 결과
    fake_loss = criterion(outputs, fake_labels)   # output과 label(0이겠죠)과 비교하여 loss 계산()
    fake_score = outputs                          # output의 리스트를 fake_score라는 변수에 저장..

    d_loss = real_loss + fake_loss                # descriminator의 loss는 두 loss를 더한 것
    d_loss.backward()                             # 오차 역전파
    d_optimizer.step()                            # parameter update
    return d_loss, real_score, fake_score


def train_generator(generator, discriminator_outputs, real_labels):
    generator.zero_grad()                                     # parameter 0으로 초기화
    g_loss = criterion(discriminator_outputs, real_labels)    # loss 계산 -> Discriminator에 fake-data를 넣었을 때 결과와
                                                              # 실제 레이블(1)을 비교, 이 부분이 핵심
    g_loss.backward()                                         # 역전파
    g_optimizer.step()                                        # parameter update
    return g_loss  
~~~
noise 벡터를 16개 먼저 뽑아두고,
Generator가 학습함에 따라 이미지 데이터가 어떻게 변화하는 지 확인해보려고 합니다.
~~~python
num_test_samples = 16
test_noise = Variable(torch.randn(num_test_samples, 100).cuda())
~~~

이제 모든 세팅은 다 끝났고, 코드를 돌려봅시다.
~~~python

# 이 부분은 이미지를 jupyter notebook에서 보여주기 위한 초기 세팅입니다

size_figure_grid = int(math.sqrt(num_test_samples))
fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(6, 6))
for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
    ax[i,j].get_xaxis().set_visible(False)
    ax[i,j].get_yaxis().set_visible(False)

# 에폭과 배치 개수를 설정합니다.

num_epochs = 200
num_batches = len(train_loader)
num_fig = 0

# Discriminator와 Generator의 Loss, 그리고 Discriminator의 real image에 대한 스코어, fake image에 대한 스코어를 tracking합시다.

tracking_dict = {}
tracking_dict["d_loss"] = []
tracking_dict["g_loss"] = []
tracking_dict["real_score"] = []
tracking_dict["fake_score"] = []

# 자 이제 진짜 시작해봅시다.

for epoch in range(num_epochs):
    for n, (images, _) in enumerate(train_loader):
        images = Variable(images.cuda())                          # real images를 pytorch 변수로 바꿔줍니다
        real_labels = Variable(torch.ones(images.size(0)).cuda()) # real images의 labels를 pytorch 변수로 바꿔줍니다.

        # 샘플링

        noise = Variable(torch.randn(images.size(0), 100).cuda()) # Generator의 인풋값인 noise를 추출한다.
        fake_images = generator(noise)                     # generator에 noise를 넣어 fake image를 만든다
        fake_labels = Variable(torch.zeros(images.size(0)).cuda())# fake images의 labels를 가져옵니다.

        # Discriminator를 학습시킵니다

        d_loss, real_score, fake_score = train_discriminator(discriminator, images, real_labels, fake_images, fake_labels)

        # Discriminator를 새로운 fake images에 대해 테스트해봅니다.

        noise = Variable(torch.randn(images.size(0), 100).cuda())
        fake_images = generator(noise)
        outputs = discriminator(fake_images)

        # 테스트 결과와 실제 레이블(영벡터..)을 비교하여 generator를 학습시킵니다.

        g_loss = train_generator(generator, outputs, real_labels)

        # 100번째 마다 test_image를 확인해볼껍니다.

        if (n+1) % 100 == 0:
            test_images = generator(test_noise)

            # 이미지를 쥬피터 노트북에 띄웁니다.

            for k in range(num_test_samples):
                i = k//4
                j = k%4
                ax[i,j].cla()
                ax[i,j].imshow(test_images[k,:].data.cpu().numpy().reshape(28, 28), cmap='Greys')
            display.clear_output(wait=True)
            display.display(plt.gcf())

            plt.savefig('results/mnist-gan-%03d.png'%num_fig)
            num_fig += 1
            print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, '
                  'D(x): %.2f, D(G(z)): %.2f'
                  %(epoch + 1, num_epochs, n+1, num_batches, d_loss.data[0], g_loss.data[0],
                    real_score.data.mean(), fake_score.data.mean()))
            tracking_dict["d_loss"].append(d_loss.data[0])
            tracking_dict["g_loss"].append(g_loss.data[0])
            tracking_dict["real_score"].append(real_score.data.mean())
            tracking_dict["fake_score"].append(fake_score.data.mean())
~~~
<br>
<br>

## Result

학습과정에서  Discriminator와 Generator의 loss는 아래와 같은 흐름을 보였습니다.

<img src="https://scontent-hkg3-2.xx.fbcdn.net/v/t1.0-9/20375755_1542404339157737_1316267578774742826_n.jpg?oh=4af5b7de8b089528413cc1bf2918cd7f&oe=5A024F41"></img>
<br>
한편, Discriminator의 정답률은 아래와 같은 추이를 보였는데요. 실제 이미지든 가짜 이미지든 학습을 진행함에 따라서 정답률이 0.5에 수렴함을 확인할 수가 있습니다.

<img src = "https://scontent-hkg3-2.xx.fbcdn.net/v/t1.0-9/20294465_1542404345824403_8046657392976649940_n.jpg?oh=e543ccd10bc8d519c66bf72c9c6fc645&oe=59FCF6F6"></img>
<br>

아래는 맨 마지막 스텝에서 생성된 이미지입니다. 약간 어설픈 결과물도 있긴 하지만 대체로 사람이 쓴 글씨같은 느낌이긴 합니다.
<img src = "https://scontent-hkg3-2.xx.fbcdn.net/v/t1.0-9/20476462_1542405222490982_2332763144594297743_n.jpg?oh=88dccb2aa99a88bed1220d50437fb571&oe=59FF28BC">

아마 단순 FC Layer 대신에 Convolution Layer를 집어넣으면 결과물이 개선될 수 있지 않을까 싶습니다. 그게 나중에 저희가 배울 DCGAN이겠죠?

아래는 Generator가 생성한 이미지들이 어떻게 변화해가는 지 보여주는 동영상입니다.

[![](https://i.ytimg.com/vi/ndhZg6gJ6bs/hqdefault.jpg?sqp=-oaymwEXCPYBEIoBSFryq4qpAwkIARUAAIhCGAE=&rs=AOn4CLCuYE1oGaBp_0CQJQIggG6SwnFLiQ)](https://www.youtube.com/watch?v=ndhZg6gJ6bs)

---
## Reference
[GAN pytorch code]
 [https://github.com/prcastro/pytorch-gan/blob/master/MNIST%20GAN.ipynb](https://github.com/prcastro/pytorch-gan/blob/master/MNIST%20GAN.ipynb)

---
