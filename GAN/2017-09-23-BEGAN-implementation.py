import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import pickle

# Settings
batchSize = 16
imageSize = 64
z_dim = 64
n_channels = 3
conv_hidden_num = 64
outf="./results"

# Import Dataset
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

# Design Discriminator in AutoEncoder-manner
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        # encoder
        self.conv1 = nn.Sequential(

            nn.Conv2d(n_channels,conv_hidden_num,3,1,1),

            nn.Conv2d(conv_hidden_num,conv_hidden_num,3,1,1),
            nn.ELU(True),
            nn.Conv2d(conv_hidden_num,2*conv_hidden_num,3,2,1),
            nn.ELU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(2*conv_hidden_num,2*conv_hidden_num,3,1,1),
            nn.ELU(True),
            nn.Conv2d(2*conv_hidden_num,3*conv_hidden_num,3,2,1),
            nn.ELU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(3*conv_hidden_num,3*conv_hidden_num,3,1,1),
            nn.ELU(True),
            nn.Conv2d(3*conv_hidden_num,3*conv_hidden_num,3,1,1),
            nn.ELU(True)

        )

        self.fc1 = nn.Linear(4*4*3*conv_hidden_num,z_dim)

        # decoder
        self.fc2 = nn.Linear(z_dim,16*16*conv_hidden_num)
        self.conv2 = nn.Sequential(

            nn.Conv2d(conv_hidden_num,conv_hidden_num,3,1,1),
            nn.ELU(True),
            nn.Conv2d(conv_hidden_num,conv_hidden_num,3,1,1),
            nn.ELU(True),
            nn.UpsamplingNearest2d(scale_factor=2),

            nn.Conv2d(conv_hidden_num,conv_hidden_num,3,1,1),
            nn.ELU(True),
            nn.Conv2d(conv_hidden_num,conv_hidden_num,3,1,1),
            nn.UpsamplingNearest2d(scale_factor=2),

            nn.Conv2d(conv_hidden_num,conv_hidden_num,3,1,1),
            nn.ELU(True),
            nn.Conv2d(conv_hidden_num,conv_hidden_num,3,1,1),
            nn.ELU(True),

            nn.Conv2d(conv_hidden_num,3,3,1,1)
        )


    def forward(self,x):

        # through encoder conv-layer
        conv = self.conv1(x)
        # embedding via encoder
        embedding = self.fc1(conv.view(-1,3*conv_hidden_num*4*4))
        # reconstructing img via decoder
        reconst = self.conv2(self.fc2(embedding).view(-1,conv_hidden_num,16,16))
        return embedding, reconst

# Design Generator - the same structure with decoder of discriminator
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.fc = nn.Linear(z_dim,16*16*conv_hidden_num)
        self.conv = nn.Sequential(

            nn.Conv2d(conv_hidden_num,conv_hidden_num,3,1,1),
            nn.ELU(True),
            nn.Conv2d(conv_hidden_num,conv_hidden_num,3,1,1),
            nn.ELU(True),
            nn.UpsamplingNearest2d(scale_factor=2),

            nn.Conv2d(conv_hidden_num,conv_hidden_num,3,1,1),
            nn.ELU(True),
            nn.Conv2d(conv_hidden_num,conv_hidden_num,3,1,1),
            nn.ELU(True),
            nn.UpsamplingNearest2d(scale_factor=2),

            nn.Conv2d(conv_hidden_num,conv_hidden_num,3,1,1),
            nn.ELU(True),
            nn.Conv2d(conv_hidden_num,conv_hidden_num,3,1,1),
            nn.ELU(True),

            nn.Conv2d(conv_hidden_num,3,3,1,1)
        )

    def forward(self,x):
        x = self.fc(x)
        x = x.view(-1,conv_hidden_num,16,16)
        out = self.conv(x)
        return out

# make instances of network
discriminator = Discriminator()
generator = Generator()

# weight initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

discriminator.apply(weights_init)
generator.apply(weights_init)

# activate cuda
discriminator.cuda()
generator.cuda()

# set optimizer and loss criterion
lr = 1e-5 # needs to be reduced in case of modal collapse
D_optimizer = optim.Adam(discriminator.parameters(),lr=lr, betas=(0.5, 0.999))
G_optimizer = optim.Adam(generator.parameters(),lr=lr, betas=(0.5, 0.999))

criterion = nn.L1Loss().cuda()

# lists to track training history
D_losses =[]
G_losses =[]
D_real_losses = []
D_fake_losses = []
measurements = []
k_ts = []
result_dict = {}

# set parameters of BEGAN
k_t = 0
gamma = 0.75
lambda_k = 0.0001


# fixed noise to check out
fixed_noise = Variable(torch.FloatTensor(batchSize*z_dim).uniform_(-1,1)).view(batchSize,z_dim).cuda()

# train
for epoch in range(50000):

    fake_ = generator(fixed_noise)
    vutils.save_image(fake_.data,"{}/generated_img_{}_epoch.png".format(outf,format(epoch,"0>5")))
    
    for step,(data,_) in enumerate(dataloader):
        n_inputs = data.size()[0]

        D_optimizer = optim.Adam(discriminator.parameters(),lr=lr, betas=(0.5, 0.999))
        G_optimizer = optim.Adam(generator.parameters(),lr=lr, betas=(0.5, 0.999))

        # gradient init as zero
        discriminator.zero_grad()
        generator.zero_grad()

        # update optimizer
        X_v = Variable(data).cuda()

        # put real-image through discriminator
        D_real_embedding, D_real_reconst  = discriminator(X_v)
        D_real_loss = criterion(D_real_reconst,X_v)

        # put fake-image through generator
        noise = Variable(torch.FloatTensor(n_inputs*z_dim).uniform_(-1,1)).view(n_inputs,z_dim).cuda()
        fake = generator(noise)

        D_fake_embedding, D_fake_reconst = discriminator(fake.detach())
        D_fake_loss_d = criterion(D_fake_reconst,fake.detach())
        D_fake_loss_g = criterion(fake,D_fake_reconst.detach())

        # calculate loss
        D_loss = D_real_loss - k_t*D_fake_loss_d
        G_loss = D_fake_loss_g

        # backprop & update network
        D_loss.backward()
        G_loss.backward()

        D_optimizer.step()
        G_optimizer.step()

        # update k_t
        k_t += lambda_k*(gamma*D_loss.data[0] - G_loss.data[0])
        k_t = max(min(k_t,1),0)

        # Calculate Convergence Measurement
        M = D_loss + torch.abs(gamma*D_loss - G_loss)

        if step%100 == 0:
            print('[%d/%d][%d/%d] D_loss: %.4f G_loss: %.4f D_real_loss: %.4f D_fake_loss: %.4f M: %.4f k: %5f'
                  % (epoch, 10000, step, len(dataloader),
                     D_loss.data[0], G_loss.data[0], D_real_loss.data[0], D_fake_loss_d.data[0],
                     M.data[0], k_t))
            # save losses and scores
            D_losses.append(D_loss.data[0])
            G_losses.append(G_loss.data[0])
            D_real_losses.append(D_real_loss.data[0])
            D_fake_losses.append(D_fake_loss_d.data[0])
            measurements.append(M.data[0])
            k_ts.append(k_t)

            result_dict["D_losses"] = D_losses
            result_dict["G_losses"] = G_losses
            result_dict["D_real_losses"] = D_real_losses
            result_dict["D_fake_losses"] = D_fake_losses
            result_dict["measurements"] = measurements
            result_dict["k_ts"] = k_ts

            pickle.dump(result_dict,open("{}/result_dict.p".format(outf),"wb"))
            if epoch<5:
                # save fixed img
                fake_ = generator(fixed_noise)
                vutils.save_image(fake_.data,"{}/generated_img_{}_epoch_{}_step.png".format(outf,format(epoch,"0>5"),step))
    if (epoch+1)%100 ==0:
        lr *= 0.955
    lr = max(lr,1e-7)
    # save model
    torch.save(discriminator.state_dict(),"D_epoch_{}.pth".format(epoch))
    torch.save(generator.state_dict(),"G_epoch_{}.pth".format(epoch))
