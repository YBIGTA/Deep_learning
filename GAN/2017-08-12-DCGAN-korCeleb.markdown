# Neuro-celeb: 인공신경망이 만드는 여자 연예인 얼굴
작성자: YBIGTA 10기 김지중

<br>
<a href="http://imgur.com/5dwOO6S"><img src="http://i.imgur.com/5dwOO6S.png" title="source: imgur.com" /></a><br>
<span style="color:#7C7877; font-family: 'Apple SD Gothic Neo'; font-weight:200">DCGAN으로 생성한 연예인 얼굴 이미지 </span>

<br>
<br>

#### 들어가며

지난 시간 우리는 Conditional GAN과 DCGAN의 개념을 정리한 후, MNIST 데이터를 통해 Conditional GAN의 예제 코드를 살펴보았습니다. 오늘은 DCGAN의 코드 구현에 대해 살펴보려고 합니다. [Pytorch 깃허브 ](https://github.com/pytorch/examples) 에 올라온 [DCGAN 예제 코드](https://github.com/pytorch/examples/tree/master/dcgan) 를 사용하였습니다. 앞에 parser 부분을 제외한다면 100줄 조금 넘는 간단한 코드입니다. 데이터 셋으로는 원래 [LFW](http://vis-www.cs.umass.edu/lfw/) 를 사용하려고 했는데요. 이름 모를 외국인 분들 이미지를 사용하는 것 보다 조금 더 재밌는 결과물을 만들어보고 싶어 한국 여자 연예인 50명의 이미지를 7,000여장 수집해보았습니다. 결과적으로 이미지 전처리하는 데 시간이 많이 걸려 모델을 제대로 씹고 뜯고 맛보지 못하지 않았나, 아쉬움이 남습니다..
<br>

---

#### 0. 데이터 수집 및 가공
먼저, 크롬 웹스토어에 있는 이미지 일괄 다운로더를 통해 대용량의 이미지를 한 번에 다운받았습니다. 얼굴이 작게 나온 사진, 마이크나 기타 물체가 얼굴을 가리고 있는 사진, 얼굴이 측면으로 나온 사진 등을 배제하기 위해 노력했습니다.(이 부분은 어쩔 수 없이 사람의 손으로 해결해야 하는 부분입니다..) 그럼에도 불구하고 측면 사진 등 적절치 못한 데이터들이 꽤 많이 저장되긴 했습니다.

이후 구글 클라우드 Vision API를 사용하여 얼굴 부분만 잘라낸 뒤 64*64 사이즈의 이미지로 저장하였습니다. 구글 클라우드 플랫폼에서는 컴퓨팅 엔진 뿐만이 아니라 Vision, Natural Language, Speech 등 여러 분야에 대한 API를 제공하고 있으며, 첫 가입 시 $300 상당의 크레딧을 제공합니다. Vision API 사용법과 관련해서는 [이 블로그](http://twinw.tistory.com/199) 를 참고하였습니다.
<br>

---


#### 1. 개발 환경

어김없이 AWS ec2를 이용하였습니다. 인스턴스는 *p2.xlarge*, AMI는 *Deep Learning AMI Ubuntu Version* 을 골랐습니다. 개발 환경의 스펙은 아래와 같습니다.

- **GPU** NVIDIA K80
- **Ubuntu** 16.04
- **CUDA** 8.0
- **Pytorch** 0.1.12

개발 환경 구축 후 본격적으로 코드를 돌리는 데 CUDA Runtime Error (30)가 발생하였습니다. 에러코드에 Unknown Error라고 써있었는데요, Pytorch [깃허브 이슈](https://www.google.co.kr/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&ved=0ahUKEwikptfUuM_VAhUKjZQKHUIzDQMQFggsMAE&url=https%3A%2F%2Fgithub.com%2Fpytorch%2Fpytorch%2Fissues%2F631&usg=AFQjCNE7s24OhFPinfGhoFZzTL61fuEKgQ) 를 보니 컴퓨터를 껐다 키면 마법처럼 다시 정상작동한다는 이야기를 보고 인스턴스를 reboot 해봤으나 소용이 없었습니다. 결국 CUDA와 Pytorch 모두 재설치했더니 문제가 해결되었습니다.
<br>

---

#### 2. 코드 분석

**1) 데이터 정의**

먼저 ImageFolder를 활용해 하위 디렉토리의 모든 이미지를 불러왔습니다. 각 RGB채널을 평균, 표준편차를 각각 0.5로 두고 표준화를 진행합니다. 대체 왜 그러는지는 아직도 의문입니다..
~~~python
des_dir = "./korCeleb64/"

dataset = dset.ImageFolder(root=des_dir,
                           transform=transforms.Compose([
                               transforms.Scale(imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size= batchSize,
                                         shuffle=True)
~~~
<br>

**2) 하이퍼 파라미터 및 상수 정의**

모델을 돌리기 위한 하이퍼 파라미터 및 상수를 정의내립니다.
~~~python
nz     = 100      # dimension of noise vector
nc     = 3        # number of channel - RGB
ngf    = 64       # generator 레이어들의 필터 개수를 조정하기 위한 값
ndf    = 64       # discriminator 레이어들의 필터 개수를 조정하기 위한 값
niter  = 200      # total number of epoch
lr     = 0.0002   # learning rate
beta1  = 0.5      # hyper parameter of Adam optimizer
ngpu   = 1        # number of using GPU

imageSize = 64    
batchSize = 64    

outf = "./celebA_result/"
~~~
<br>

**3) 파라미터 초기화**

밑에서 확인하겠지만, DCGAN의 레이어에는 Convolution(혹은 Convolution transpose), Batch Normalization 두 가지 종류가 있습니다. 각각 레이어를 어떻게 초기화 할 지 함수를 통해 정의합니다.
~~~python
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:         # Conv weight init
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:  # BatchNorm weight init
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
~~~
<br>

**4) 모델링**

먼저 Generator 모델입니다.
<a href="http://imgur.com/H61p6qd"><img src="http://i.imgur.com/H61p6qd.png" title="source: imgur.com" /></a>
출처: [DCGAN in Tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)

김태훈님의 깃허브에서 이미지를 가져왔습니다. 필터 크기나 stride값 등 아래 코드와 다른 점이 있지만, Transposed Convolution 과정이 잘 표현된 것 같아 가져왔습니다.

ConvTranspose2d의 인자는 input 채널 수, output 채널 수, 필터 사이즈, stride, padding의 순서입니다.

~~~python
class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(

            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()

            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
~~~
<br>


그 다음으로는 Discrimnator입니다. Discriminator는 일반적인 CNN과 유사합니다.
* RGB값을 갖는 64*64의 이미지를 input으로 받습니다
* 각 레이어의 channel 수는 Generator에서의 순서를 거꾸로 한다.
* 5번의 convolution 끝에 한 개의 스칼라값을 갖게됩니다.
  * sigmoid를 통해 하나의 확률값으로 변환시킵니다.
~~~python
class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()

            # state size. 1
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)
~~~
<br>
generator와 discriminator 인스턴스를 생성하고 파라미터를 초기화합니다.

~~~python
netG = _netG(ngpu)
netG.apply(weights_init)
print(netG)

netD = _netD(ngpu)
netD.apply(weights_init)
print(netD)
~~~
<br>

loss는 binary classification error를 사용합니다. 그리고 batchSize 크기의 빈 Tensor를 미리 만들어 둡니다. 나중에 여기에 실제 혹은 가짜 label을 집어 넣을 건데요, TensorFlow의 placeholder 쓰는거랑 비슷한 느낌인 것 같습니다. 그리고 주요 변수가 cuda를 활용할 수 있게 cuda를 활성화시켜줍니다.
~~~python
criterion = nn.BCELoss()

input = torch.FloatTensor(batchSize, 3, imageSize,imageSize)
noise = torch.FloatTensor(batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(batchSize, nz, 1, 1).normal_(0, 1)

label = torch.FloatTensor(batchSize)
real_label = 1
fake_label = 0

netD.cuda()
netG.cuda()
criterion.cuda()
input, label = input.cuda(), label.cuda()
noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
~~~
<br>

학습 진행과정에 따라 결과 이미지가 어떻게 달라지는 지 관찰하기 위해 두고두고 살펴볼 고정된 noise값을 지정합니다. optimizer는 Adam을 사용합니다.
~~~python
fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

result_dict = {}
loss_D,loss_G,score_D,score_G1,score_G2 = [],[],[],[],[]
~~~
<br>

학습 과정은 지난 번과 마찬가지로 아래와 같습니다.

**Discriminator 학습**

>(1) 실제 이미지들을 넣고 분류기를 돌려봅니다.
>* real_loss: 실제 이미지들을 넣은 결과값들(0 혹은 1로 구성된 벡터)와 실제 이미지들의 레이블(1로 이루어진 벡터)를 비교해서 계산된 loss<br>
>
>(2) fake 이미지들을 넣고 분류기를 돌려봅니다.
>  * fake_loss: fake 이미지들을 넣은 결과값(0 혹은 1로 구성된 벡터)와 fake 이미지들의 레이블(영벡터)를 비교해서 계산된 loss<br>
>
>(3) Discriminator's loss =  real_loss + fake_loss<br>
>
>(4) 오차 역전파 및 파라미터 업데이트<br>

<br>

**Generator 학습**

>(1) 새로운 fake 이미지들을 뽑아서 Discriminator에 일종의 테스트 셋으로 넣어봅니다.
>  * fake 이미지는 역시 Generator에 noise를 넣어서 만듭니다.<br>
>
>(2) 테스트 결과값과 실제 이미지의 레이블을 비교해 loss를 계산합니다.<br>
>
>(3) 오차 역전파 및 파라미터 업데이트<br>

<br>

~~~python
for epoch in range(niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        # train with real
        netD.zero_grad()
        real_cpu, _ = data
        batch_size = real_cpu.size(0)

        real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)

        inputv = Variable(input)
        labelv = Variable(label)

        output = netD(inputv)
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        labelv = Variable(label.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()

        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        output = netD(fake)

        errG = criterion(output, labelv)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()



    vutils.save_image(real_cpu,
            '%s/real_samples.png' % outf,
            normalize=True)
    fake = netG(fixed_noise)
    vutils.save_image(fake.data,
            '%s/fake_samples_epoch_%03d.png' % (outf, epoch),
            normalize=True)
    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
      % (epoch, niter, i, len(dataloader),
         errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
    loss_D.append(errD.data[0])
    loss_G.append(errG.data[0])
    score_D.append(D_x)
    score_G1.append(D_G_z1)
    score_G2.append(D_G_z2)
    result_dict = {"loss_D":loss_D,"loss_G":loss_G,"score_D":score_D,"score_G1":score_G1,"score_G2":score_G2}
    pickle.dump(result_dict,open("./{}/result_dict.p".format(outf),"wb"))

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG.pth' % (outf))
    torch.save(netD.state_dict(), '%s/netD.pth' % (outf))
~~~
---

#### 3. 결과 분석

**3-1. Loss 및 Score**

전반적으로, 지난 번 MNIST 처럼 이쁘게 수렴하는 양상을 보이지는 않았습니다. 앞으로 그렇게 이쁘게 수렴하는 경우는 없을 것 같네요.. 우선 generator의 loss가 하향곡선을 그리지 못하고 있습니다. 결국 실제 데이터와 구분이 쉬운 fake 데이터를 만들게 되고, 따라서 discriminator의 loss는 계속 낮은 수준을 유지했습니다. 이건 데이터가 적었기 때문이라고 해석할 수도 있는데요, GAN 구현 왕 김태훈님의 [깃허브](https://github.com/carpedm20/DCGAN-tensorflow)를 보니 원래 loss가 그렇게 쉽게 수렴하는 거 같지는 않습니다. 다만 그걸 감안하더라도 generator의 loss가 좀 많이 높네요.

<a href="http://imgur.com/e6i6xf0"><img src="http://i.imgur.com/e6i6xf0.png" title="source: imgur.com" /></a>
<br>

Discriminator에게 이미지를 넣어서 학습시키면 이 이미지가 진짜일 확률을 뱉어냅니다. 아래 그래프에서 D(x)는 실제 이미지에 대한 확률값으로, 떨어지는 구간도 자주 보이지만 대체로 1 근처에 위치합니다. 반면 D(G(z))는 가짜 이미지를 넣었을 때 나온 결과입니다. 이 곡선은 그래프 주로 아래쪽에 위치하고 있는데요, 결국 discriminator가 가짜 이미지든 진짜 이미지든 잘 구별하고 있다는 것을 보여줍니다.

<a href="http://imgur.com/YlkVCBV"><img src="http://i.imgur.com/YlkVCBV.png" title="source: imgur.com" /></a>
<br>

**3-2. 결과 이미지**

학습 과정을 영상으로 남겨보았습니다. 만들어지는 이미지가 어떤 과정으로 변했는 지 확인하고 싶으시다면 아래 영상을 시청하시면 됩니다.

[![](https://i.ytimg.com/vi/xbKQPNGfuHY/hqdefault.jpg)](https://youtu.be/xbKQPNGfuHY)

200 에폭씩 학습을 진행했는데도 눈코입이 제 자리를 찾지 못한다거나 전반적인 얼굴 형태가 아예 잡히지 않는 이미지들이 상당히 많습니다. 이는 데이터 양이 절대적으로 부족하며, 그나마 있는 데이터들의 퀄리티가 좋지 않기 때문으로 생각됩니다. 퀄리티가 좋지 않다는 것은 사람의 얼굴 각도가 이미지별로 천차만별이며, 측면에서 찍힌 사진, 눈 감고 찍힌 사진 등 부적절한 이미지들이 꽤 남아있다는 점에서 비롯됩니다. 괜히 연구자들이 검증된 데이터셋을 사용하는 게 아니었군요. (문서 맨 처음에 업로드 된 이미지는 괜찮은 것만 모아놓은 이미지였습니다..) Latent Space Analysis도 진행해보고 싶었으나, 적절한 주제가 떠오르지 않아 생략하였습니다. 모델이 생성한 다른 이미지들을 업로드하며 문서를 마치고자 합니다.

<a href="http://imgur.com/yuyLcIy"><img src="http://i.imgur.com/yuyLcIy.png" title="source: imgur.com" /></a>

<span style="color:#7C7877; font-family: 'Apple SD Gothic Neo'; font-weight:200">Neuro-celeb 200 epoch example1 </span>

<a href="http://imgur.com/0msShNI"><img src="http://i.imgur.com/0msShNI.png" title="source: imgur.com" /></a>

<span style="color:#7C7877; font-family: 'Apple SD Gothic Neo'; font-weight:200">Neuro-celeb 200 epoch example2 </span>

<a href="http://imgur.com/cWYWUJh"><img src="http://i.imgur.com/cWYWUJh.png" title="source: imgur.com" /></a>

<span style="color:#7C7877; font-family: 'Apple SD Gothic Neo'; font-weight:200">CelebA mini-set (15000 imgs) 200 epoch example2 </span>
