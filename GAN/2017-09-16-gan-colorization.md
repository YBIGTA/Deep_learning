---
layout: post
title: GAN으로 흑백 이미지에 색깔입히기
excerpt: 흑백 이미지에 색깔을 입히도록 GAN을 학습시켜봤습니다.
categories: [GAN]
comments: true
use_math: true
---

# GAN으로 흑백 이미지 색칠해보기
<h6 align="right">강병규</h6>

이번에는 GAN을 사용해서 흑백 이미지를 색칠해보려고합니다. Pytorch를 사용할거구요.

## Dataset

데이터로는 Pytorch의 torchvision안에 있는 여러 데이터셋 중에서 STL10을 사용해보려고 합니다. CIFAR-10과 유사한 데이터라고 생각하시면 되는데요. CIFAR-10은 32 x 32의 이미지들이었던 반면에, [STL10](https://cs.stanford.edu/~acoates/stl10/)은 더 고해상도의 96 x 96의 이미지를 지원합니다.

## Architecture

주된 구조는 Github을 참조했습니다.
[github gan colorizer](https://github.com/aleju/colorizer)

위의 주소를 보면 ${G}$에 대해 상세하게 설명해주고 있습니다.

![g](https://user-images.githubusercontent.com/25279765/30491122-2df33036-9a77-11e7-9f0e-17573285527a.png)

사진은 위의 링크에서 가져왔습니다. ${G}$의 구조인데요, 흑백의 이미지를 노이즈와 결합시켜줍니다. 이때 이미지의 사이즈는 64 x 64로 리사이즈해줬습니다. 거기에 흑백이니 이미지 데이터는 1 x 64 x 64의 배열입니다. 여기에 noise 또한 1 x 64 x 64의 크기를 갖게하고, 0~1사이의 uniform distribution에서 샘플링한다음, 이것과 흑백이미지를 합쳐줍니다. 즉 2 x 64 x 64가 되는 것이죠. 다시 concat을 하기전까지의 네트워크는 쉽게 생각하면 encoder라고 할 수 있습니다. 이미지의 정보를 압축하는 역할을 수행하는 것입니다. 커널의 크기는 3 x 3, stride와 padding은 1이므로, Convolution layer를 거쳐도 데이터의 크기는 변화하지않습니다.

중간에 2 x 2 max pooling을 두 번 거치면서 이미지의 크기는 4분의 1로 줄어들게 됩니다. 이후 Upsample을 다시 두번 해줘 이미지의 크기를 원상태로 돌린다음, 이상태에서 네트워크의 맨 처음 입력이었던 흑백+노이즈를 다시 합쳐줍니다. 이렇게 해서 encoder가 이미지의 정보를 압축하는데 집중할 수 있게 했다고 주장하는데 잘모르겠습니다...

${D}$의 구조는 DCGAN에서 사용했던 것과 동일합니다. 위의 깃헙에서는 일반적인 CNN 네트워크에 Fully Connected를 쌓았다고 하는데, 저는 그렇게 하지 않았습니다.

## Implementation

이때 STL10은 컬러 이미지임에 주의해야합니다. 즉 ${G}$에 입력으로 넣어줘야할때, 흑백으로 처리해주어야 한다는 뜻이지요. 여러 방법을 찾아봤지만 Pytorch의 경우 Tensor를 사용하기에 마땅하지 않았습니다. 그래서 그냥 Tensor를 Numpy array로 바꾼다음 일반적인 공식을 적용했습니다.

$${ G = 0.2989 * R + 0.5870 * G  + 0.1140 * B} $$

코드로 구현하면 다음과 같아집니다.

```python
def toGray(rgb, batchsize):
    """shape of rgb is bs x 3 x 64 x 64, order by RGB"""
    lst = []
    for data in rgb:
        r, g, b = data[0, :, :], data[1, :, :], data[2, :, :]
        gray = 0.2989 *r + 0.5870 * g + 0.1140 * b
        lst.append(gray)
    return np.asarray(lst).reshape(batchsize, 1, 64, 64) # (bs, 64, 64)
```

3 x 64 x 64의 이미지가 배치사이즈만큼 들어있는데요, 리스트를 하나 만든다음 흑백이미지를 이 리스트에 넣어줍니다.

### Discriminator

아까 위에서 ${D}$의 경우 DCGAN을 그대로 사용한다했는데요, 학습이 원활하게 진행되지 않고 그냥 바로 검은색으로 수렴해버렸습니다. 그래서 Conditional GAN의 느낌으로 흑백이미지를 진짜/가짜 컬러 이미지와 합쳐서 ${D}$에 넣어주었더니 학습이 진행되었습니다.

${D}$의 구조는 다음과 같습니다.

```python
class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()
        self.main = nn.Sequential(
            # 3+1 x 64 x 64
            nn.Conv2d(nc+1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # ndf x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf * 2) x 16 x16
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf * 4) x 8 x 8
            nn.Conv2d(ndf *4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf * 8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()

        )
    def forward(self, input):
        output = self.main(input)
        return output.view(-1,1).squeeze(1)
```

${D}$는 이미지가 진짜인지 가짜인지 확률값 하나 만을 만들어냅니다.

### G

그다음으로 G를 정의할 차례입니다. 이미지를 인코딩하는 과정과 컬러를 입히는 과정을 분리했습니다.
```python
class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        self.before = nn.Sequential(
            # input : (bs x (1+1) x 64 x 64), 노이즈와 합쳐줍니다.
            nn.Conv2d(1+1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),

            # bs x 16 x 64 x 64
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),

            # bs x 32 x 32 x 32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),

            # bs x 64 x 16 x 16
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),

            # bs x 128 x 16 x 16
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),

            # bs x 256 x 16 x 16
            nn.UpsamplingNearest2d(scale_factor=2),
            # bs x 256 x 32 x32
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),

            # bs x 128 x 32 x 32
            nn.UpsamplingNearest2d(scale_factor=2),
            # bs x 128 x 64 x 64
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
            # bs x 64 x 64 x 64
        )
        self.after = nn.Sequential(
            # 흑백이미지와 노이즈를 다시 합쳐줍니다
            # bs x (64 + 2) x 64 x 64
            nn.Conv2d(64 + 2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # bs x 32 x 64 x 64
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
            # bs x 3 x 64 x 64
        )

    def forward(self, input):
        encode = self.before(input)
        decode = torch.cat([input, encode], dim=1)
        output = self.after(decode)
        return output
```

weight를 초기화해주는 코드입니다.
```python
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:         # Conv weight init
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:  # BatchNorm weight init
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
```

네트워크들을 선언하고 weight를 초기화해줍니다.
```python
netG = _netG()
netG.apply(weights_init)
print(netG)

netD = _netD()
netD.apply(weights_init)
print(netD)
```

그 다음에는 loss와 optimizer를 선언해줍니다.
```python
criterion = nn.BCELoss()

input = torch.FloatTensor(batchSize, 3, 64, 64)
noise = torch.FloatTensor(batchSize, 1, 64, 64)

label = torch.FloatTensor(batchSize)
real_label = 1
fake_label = 0
optimizerD = optim.Adam(netD.parameters(), lr=0.0002,betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002,betas=(0.5, 0.999))
result_dict= {}
loss_D, loss_G = [], []
```

```python
niter=200
for epoch in range(niter):
    for i, (data, _) in enumerate(train_loader):
        #############
        # D!
        #############
        batchSize = len(data)
        # real
        netD.zero_grad()
        value = data
        # 흑백 이미지
        gray_array = toGray(value.numpy(), batchSize)

        input.resize_as_(value).copy_(value)
        label.resize_(len(data)).fill_(real_label)

        inputv = Variable(torch.from_numpy( np.concatenate((input.numpy(), gray_array), axis=1) ))
        labelv = Variable(label)

        output = netD(inputv)
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        # Fake
        noise.resize_(batchSize, nz_C, nz_H, nz_W).uniform_(0,1) # 64 x 1 x 64 x 64
        # 노이즈와 섞어줍니다
        gray_noise = np.concatenate((gray_array, noise.numpy()), axis=1)

        gray_noisev = Variable(torch.from_numpy(gray_noise))

        fake = netG(gray_noisev)

        fakev = Variable(torch.from_numpy(np.concatenate((fake.data.numpy(), gray_array), axis=1)))
        labelv = Variable(label.fill_(fake_label))
        output = netD(fakev)
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()

        errD = errD_real + errD_fake
        optimizerD.step()

        ##############
        # G!
        ##############
        netG.zero_grad()
        labelv = Variable(label.fill_(real_label))
        output = netD(fakev)

        errG = criterion(output, labelv)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()

        if ((i+1) % 100 == 0):
            print(i, "step")
            print(D_x)
            print(D_G_z1)
            if not os.path.exists('results/'):
                os.makedirs('results/')
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%s.png' % (outf, str(epoch)+" "+str(i+1)))
    vutils.save_image(fake.data,'%s/fake_samples_epoch_%s.png' % (outf, epoch),normalize=True)
    loss_D.append(errD.data[0])
    loss_G.append(errG.data[0])
    result_dict = {"loss_D":loss_D,"loss_G":loss_G}
    pickle.dump(result_dict,open("./{}/result_dict.p".format(outf),"wb"))
    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG.pth' % (outf))
    torch.save(netD.state_dict(), '%s/netD.pth' % (outf))
```

전체적인 학습과정입니다. ${D}$를 먼저 학습시키는데요, 진짜 이미지와 이를 흑백처리한 이미지를 같이 넣어준다음 loss를 구하고, 가짜이미지와 흑백이미지를 concat한다음 다시 loss를 구해 이를 역전파합니다.

${G}$의 경우에는 이미지를 진짜같이 만드는 것이 목적이기에 criterion에 real_label, 1을 넣어서 학습을 시켜줍니다.

## Result

To be added

## Reference

https://github.com/aleju/colorizer

http://cs231n.stanford.edu/reports/2017/pdfs/302.pdf
