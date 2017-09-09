
# 필요한 모듈 불러오기


```python
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F

from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
% matplotlib inline

import os
import random
import numpy as np
import pickle
```

# 데이터 불러오기


```python
transform = transforms.Compose([          
        transforms.ToTensor(),                     
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

cudnn.benchmark = True

train_dataset = dsets.MNIST(root='./data/', train=True, download=True, transform=transform) 
data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
```


```python
class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        
        self.fc = nn.Sequential(
                
                    nn.Linear(74,1024),
                    nn.BatchNorm1d(1024),
                    nn.LeakyReLU(0.1),
                    nn.Linear(1024,7*7*128),
                    nn.BatchNorm1d(7*7*128),
                    nn.LeakyReLU(0.1)

                )
        
        self.conv = nn.Sequential(
            
                    nn.ConvTranspose2d(128, 64, kernel_size=4,stride=2,padding=1),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.1),
                    nn.ConvTranspose2d(64, 1, kernel_size=4,stride=2,padding=1),

        )


        
    def forward(self, x):
        
        x = self.fc(x)
        x = x.view(batchSize,128,7,7)
        x = self.conv(x)

        return x
```

Hout=(Hin−1)∗stride[0]−2∗padding[0]+kernel_size[0]+output_padding[0]


```python
class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()
        self.conv = nn.Sequential(
                    
                    nn.Conv2d(1, 64, kernel_size=4,stride=2,padding=1),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(64, 128, kernel_size=4,stride=2,padding=1),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.1)
        
        )
        
        self.fc1 = nn.Sequential(
        
                    nn.Linear(6272, 1024),
                    nn.BatchNorm1d(1024),
                    nn.LeakyReLU(0.1),
        )

        self.fc2 = nn.Linear(1024,1)
        
    def forward(self, x):

        x = self.conv(x)
        x = self.fc1(x.view(batchSize,-1))
        out_fc = self.fc2(x)
        out_proba = F.sigmoid(out_logit)
        
        return out_proba, out_fc, x
```


```python
class _netQ(nn.Module):
    def __init__(self):
        super(_netQ, self).__init__()

        self.fc1  = nn.Sequential(
                    
                    nn.Linear(1024,64),
                    nn.BatchNorm1d(64),
                    nn.LeakyReLU(0.1)
        )
        
        self.fc2 = nn.Linear(64,12)

    def forward(self, x):
        
        x = self.fc1(x)
        out_fc = self.fc2(x)
        out_proba = F.softmax(out_logit)

        return out_proba, out_fc
```

# 파라미터 초기값 설정


```python
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
```

# 네트워크 초기화


```python
netG = _netG()
netG.apply(weights_init)

print(netG)
```


```python
netD = _netD()
netD.apply(weights_init)

print(netD)
```


```python
netQ = _netQ()
netQ.apply(weights_init)

print(netQ)
```

# 기타 다른 초기값 세팅


```python
niter = 200
batchSize = 100
imageSize = 28
nz = 62

input = torch.FloatTensor(batchSize, 3, imageSize,imageSize)
noise = torch.FloatTensor(batchSize, nz, 1, 1)

label = torch.FloatTensor(batchSize,1).cuda()
real_label = 1.0
fake_label = 0.0
```


```python
noise = torch.FloatTensor(batchSize, nz, 1, 1)
```


```python
input = input.cuda()
```


```python
netQ.cuda()
netD.cuda()
netG.cuda()
```

# Loss 기준 및 Optimizer


```python
import itertools
```


```python
criterion1 = nn.BCELoss().cuda()
criterion2 = nn.CrossEntropyLoss().cuda()
criterion3 = nn.MSELoss().cuda()

beta1 = 0.5

optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=1e-3, betas=(beta1, 0.999))
optimizerQ = optim.Adam(itertools.chain(netD.parameters(),netG.parameters(),netQ.parameters()), lr=1e-3, betas=(beta1, 0.999))
```

# Sampling 함수들


```python
def sample_disc(size):
    sampleClass = np.random.randint(0,10,100).astype(int)
    sample = np.zeros((100,10))
    for i in range(100):
        sample[i][sampleClass[i]-1] = 1
    return sampleClass, sample
```


```python
def sample_cont(size):
    return np.random.uniform(-1,1,size=size).astype(float)
```

# loss / score 담을 변수


```python
result_dict = {}
loss_D = []
loss_G = []
loss_Q = []
score_D = []
score_G = []
```


```python
niter = 200


for epoch in range(niter):
    for i, data in enumerate(tqdm_notebook(data_loader)):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        real_cpu = real_cpu.cuda()
        
        inputv = Variable(real_cpu)
        labelv = Variable(label.fill_(real_label))
        output,_,_ = netD(inputv)

        errD_real = criterion1(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()
        
        # train with fake
        noise.resize_(batchSize, nz).normal_(0, 1)
        
        c_disc_class,c1 = sample_disc(batchSize)
        c_disc_class = torch.LongTensor(c_disc_class)
        c1 = torch.FloatTensor(c1)
        c2 = sample_cont(batchSize).reshape(batchSize,1)
        c2 = torch.FloatTensor(c2)
        c3 = sample_cont(batchSize).reshape(batchSize,1)
        c3 = torch.FloatTensor(c3)
        
        c_disc = c1.type(torch.FloatTensor)
        c_cont = torch.cat([c2,c3],1)
        c = torch.cat([c_disc,c_cont],1)
        noisev = Variable(torch.cat([noise,c],1).cuda())

        fake = netG(noisev)
        labelv = Variable(label.fill_(fake_label))
        output,_,_ = netD(fake.detach())
        errD_fake = criterion1(output, labelv)
        D_G_z = output.data.mean()
        errD_fake.backward()
        errD = errD_fake + errD_real
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ############################
        
        netG.zero_grad()
        labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        output,_,_ = netD(fake)
        errG = criterion1(output, labelv)
        errG.backward()
        optimizerG.step()
        
        
        ############################
        # (3) Update G network again to ensure that errD doesn't go zero
        ############################
        
        fake = netG(noisev)
        netG.zero_grad()
        labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        output,_,_ = netD(fake)
        errG = criterion1(output, labelv)
        errG.backward()
        optimizerG.step()

         
        ############################
        # (4) Update Q network       
        ############################
        fake = netG(noisev)
        _,_,output = netD(fake)
        c_logit_fake, c_fake = netQ(output)        
        netQ.zero_grad()
        netD.zero_grad()
        netG.zero_grad()

        c_disc_est = c_logit_fake[:, :10]
        q_disc_loss = criterion2(c_disc_est,Variable(c_disc_class.cuda()))
        
        c_cont_est = c_fake[:, -2:]
        q_cont_loss = criterion3(c_cont_est,Variable(c_cont.cuda()))
        
        q_loss = q_disc_loss + q_cont_loss
        q_loss.backward()
        optimizerQ.step()
        
        if i % 100 == 0:
            loss_D.append(errD.data[0])
            loss_G.append(errG.data[0])
            loss_Q.append(q_loss.data[0])
            score_D.append(D_x)
            score_G.append(D_G_z)
            result_dict = {"loss_D":loss_D,"loss_G":loss_G,"loss_Q":loss_Q,"score_D":score_D,"score_G":score_G}
            print("loss_d:{}, loss_G:{}, loss_Q:{}, score_D:{}, score_G:{}".format(errD.data[0],errG.data[0],q_loss.data[0],D_x,D_G_z))
            pickle.dump(result_dict,open("result_dict.p","wb"))
            
            # save imgs
            samples = netG(noisev).data.cpu().numpy()[:16]
            fig = plt.figure(figsize=(4, 4))
            gs = gridspec.GridSpec(4, 4)
            gs.update(wspace=0.05, hspace=0.05)
            for j, sample in enumerate(samples):
                ax = plt.subplot(gs[j])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
            fig.savefig("result/test_imgs_{}_{}.png".format(epoch,i))
    # do checkpointing
    torch.save(netG.state_dict(), 'netG.pth')
    torch.save(netD.state_dict(), 'netD.pth')
    torch.save(netQ.state_dict(), 'netQ.pth')    
```

### Result - Loss

<a href="https://imgur.com/yo1tFNI"><img src="https://i.imgur.com/yo1tFNI.png" title="source: imgur.com" /></a>

### Result - scores of Discriminator

<a href="https://imgur.com/Oua61Pe"><img src="https://i.imgur.com/Oua61Pe.png" title="source: imgur.com" /></a>

### Class Label Variation - Variation with c1(discrete code)

<a href="https://imgur.com/zbCEYuG"><img src="https://i.imgur.com/zbCEYuG.png" title="source: imgur.com" /></a>

### Unrecognizable Latent Code Variation - Variation with c2(continuous code)

<a href="https://imgur.com/K6VXoPy"><img src="https://i.imgur.com/K6VXoPy.png" title="source: imgur.com" /></a>

### Slope Variation - Variation with c3(continuous code)

<a href="https://imgur.com/KuzCXPj"><img src="https://i.imgur.com/KuzCXPj.png" title="source: imgur.com" /></a>
