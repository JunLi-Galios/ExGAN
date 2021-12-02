from tensorboardX import SummaryWriter
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
import torch.optim as optim
from torch import LongTensor, FloatTensor
from scipy.stats import skewnorm, genpareto
from torchvision.utils import save_image
import sys
from Extremeness import AvgExtremeness, MaxExtremeness


class NWSDataset(Dataset):
    """
    NWS Dataset
    """

    def __init__(
        self, path='/mnt/home/junli/PGGAN/data/', dsize=2556
    ):
        self.real = torch.load(path+'real.pt').cuda()
        self.indices = np.random.permutation(dsize)
        self.real.requires_grad = False
        
    def __len__(self):
        return self.real.shape[0]

    def __getitem__(self, item):
        return self.real[self.indices[item]]

dataloader = DataLoader(NWSDataset(), batch_size=256, shuffle=True)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


def convTBNReLU(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(0.2, True),
    )


def convBNReLU(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(0.2, True),
    )


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block1 = convTBNReLU(in_channels, 512, 4, 1, 0)
        self.block2 = convTBNReLU(512, 256)
        self.block3 = convTBNReLU(256, 128)
        self.block4 = convTBNReLU(128, 64)
        self.block5 = nn.ConvTranspose2d(64, out_channels, 4, 2, 1)

    def forward(self, inp):
        out = self.block1(inp)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        return torch.tanh(self.block5(out))


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.block1 = convBNReLU(self.in_channels, 64)
        self.block2 = convBNReLU(64, 128)
        self.block3 = convBNReLU(128, 256)
        self.block4 = convBNReLU(256, 512)
        self.block5 = nn.Conv2d(512, 64, 4, 1, 0)
        self.source = nn.Linear(64, 1)

    def forward(self, inp):
#         print('D input size', inp.size())
#         print('D max', inp.max())
#         print('D min', inp.min())
        out = self.block1(inp) 
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        size = out.shape[0]
        out = out.view(size, -1)
        source = torch.sigmoid(self.source(out))
        return source
    
class Aggregator(nn.Module):
    def __init__(self, in_channels, out_mu, out_channels):
        super(Aggregator, self).__init__()
        self.in_channels = in_channels
        self.block1 = convBNReLU(self.in_channels, 64)
        self.block2 = convBNReLU(64, 128)
        self.block3 = convBNReLU(128, 256)
        self.block4 = convBNReLU(256, 512)
        self.block5 = nn.Conv2d(512, 64, 4, 1, 0)
        self.mu = nn.Linear(64, out_mu)
        self.sigma = nn.Linear(64, out_channels)
        self.gamma = nn.Linear(64, out_channels)
        self.out_channels = out_channels

    def forward(self, inp):
        out = self.block1(inp) 
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        size = out.shape[0]
        print('out', out.size())
        out = out.view(size, -1)
        print('out', out.size())
        mu = F.softplus(self.mu(out))
        sigma = self.sigma(out)
        gamma = self.gamma(out)
        return mu, sigma, gamma
      
class Transformer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transformer, self).__init__()
        self.T_sigma = nn.Linear(in_channels, out_channels)
        self.T_gamma = nn.Linear(in_channels, out_channels)

    def forward(self, sigma, gamma):
        sigma_out = self.T_sigma(sigma)
        gamma_out = self.T_gamma(gamma)
        return F.softplus(sigma_out), F.softplus(gamma_out)

    
avg_e = AvgExtremeness()
max_e = MaxExtremeness() 
e_list = [avg_e, max_e]
      
latentdim = 20
img_size = [64, 64]
criterionSource = nn.BCELoss()
criterionContinuous = nn.L1Loss()
criterionValG = nn.L1Loss()
criterionValD = nn.L1Loss()
k = 2
G = Generator(in_channels=latentdim, out_channels=1).cuda()
D = Discriminator(in_channels=1).cuda()
A = Aggregator(1, 2, out_channels=latentdim).cuda()
T = Transformer(latentdim, np.prod(img_size)).cuda()
G.apply(weights_init_normal)
D.apply(weights_init_normal)
A.apply(weights_init_normal)
T.apply(weights_init_normal)



optimizerG = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizerA = optim.Adam(A.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizerT = optim.Adam(T.parameters(), lr=0.0001, betas=(0.5, 0.999))

static_z = Variable(FloatTensor(torch.randn((81, latentdim, 1, 1)))).cuda()

def sample_image(batches_done):
    static_sample = G(static_z).detach().cpu()
    static_sample = (static_sample + 1) / 2.0
    save_image(static_sample, DIRNAME + "%d.png" % batches_done, nrow=9)

DIRNAME = 'PGGAN/'
os.makedirs(DIRNAME, exist_ok=True)

board = SummaryWriter(log_dir=DIRNAME)

def pick_samples(samples, e_list, mu):
    batch = samples.size()[0]
    flag_list = []
    for i in range(len(e_list)):
        e = e_list[i]
        flag = e.cal_extreme(samples) > mu[i]
        flag_list.append(flag)

#     flags = torch.stack(flag_list, dim=1)
# #     print('flags', flags.size())
    
#     flags.max()
    total_flag = flag_list[0]
    
    for i in range(len(flag_list)):
        total_flag = total_flag | flag_list[i]
        
    print(total_flag)
    extremes = samples[total_flag]
    print('extremes size', extremes.size())
    
    return extremes

step = 0
ratio = 0.001
# mu = torch.ones(img_size)
mu = torch.ones(k).cuda() * 0.5
sigma = torch.ones(img_size).cuda()
gamma = torch.ones(img_size).cuda()
expo = torch.distributions.exponential.Exponential(torch.ones([1] + img_size))
n_extremes_list = []
acc_list = []


def cal_mu_incre(e_list, mu):
    min_mu_incre = np.inf
    for i in range(len(e_list)):
        e = e_list[i]
        mu_incre = e.optimize(img_size, mu[i])
        if min_mu_incre > mu_incre:
            min_mu_incre = mu_incre
            
    return min_mu_incre


for epoch in range(1000):
    print(epoch)
    for images in dataloader:
        mu_val, sigma_val, gamma_val = A(images)
        sigma_val, gamma_val = T(sigma_val, gamma_val)
        print('mu_val size', mu_val.size())
        mu_incre = torch.mean(torch.abs(mu_val),dim=0)
        print('mu_incre size', mu_incre.size())
        mu = (1 - ratio) * mu + ratio * mu_incre
        
        sigma_incre = torch.mean(sigma_val,dim=0)
        sigma_incre = sigma_incre.reshape(img_size)
        print('sigma_incre size', sigma_incre.size())
        sigma = (1 - ratio) * sigma + ratio * sigma_incre
        
        gamma_incre = torch.mean(gamma_val,dim=0)
        gamma_incre = gamma_incre.reshape(img_size)
        print('gamma_incre size', gamma_incre.size())
        gamma = (1 - ratio) * gamma + ratio * gamma_incre
        
        extreme_samples = pick_samples(images, e_list, mu)
        
        min_mu_incre = cal_mu_incre(e_list, mu)
        print('min_mu_incre size', min_mu_incre.size())
        print('min_mu_incre', min_mu_incre)
        
        extreme_samples = extreme_samples - min_mu_incre
        
        n_extremes = len(extreme_samples)
        print('n_extremes', n_extremes)
        n_extremes_list.append(n_extremes)    
        
        noise = 1e-5*max(1 - (epoch/500.0), 0)
        step += 1
        batch_size = images[0].shape[0]
        trueTensor = 0.7+0.5*torch.rand(batch_size)
        falseTensor = 0.3*torch.rand(batch_size)
        probFlip = torch.rand(batch_size) < 0.05
        probFlip = probFlip.float()
        trueTensor, falseTensor = (
            probFlip * falseTensor + (1 - probFlip) * trueTensor,
            probFlip * trueTensor + (1 - probFlip) * falseTensor,
        )
        trueTensor = trueTensor.view(-1, 1).cuda()
        falseTensor = falseTensor.view(-1, 1).cuda()
        extreme_samples = extreme_samples.cuda()
        print('trueTensor', trueTensor.size())
        realSource = D(extreme_samples + noise*torch.randn_like(extreme_samples).cuda())
        realLoss = criterionSource(realSource, trueTensor.expand_as(realSource))
        print('realSource', realSource.size())        
        
        latent = Variable(torch.randn(n_extremes, latentdim, 1, 1)).cuda()
        
        fakeData = G(latent)
        print('fakeData', fakeData.size())
        max_value, _ = torch.max(torch.reshape(fakeData, [n_extremes, -1]), dim=1)
        max_value = torch.reshape(max_value, [-1, 1, 1, 1])
        print('max_value', max_value.size())
        G_samples = fakeData - max_value
        expo_samples = expo.rsample([len(G_samples)]).cuda()
        print('expo_samples', expo_samples.size())
        G_extremes = sigma * (G_samples + expo_samples)
        
        print('images', images.size())
        print('G_extremes', G_extremes.size())
        
        fakeSource = D(G_extremes.detach())
        fakeLoss = criterionSource(fakeSource, falseTensor.expand_as(fakeSource))
        lossD = realLoss + fakeLoss
        optimizerD.zero_grad()
        lossD.backward()
        torch.nn.utils.clip_grad_norm_(D.parameters(),20)
        optimizerD.step()
        
        fakeSource = D(G_extremes)
        trueTensor = 0.9*torch.ones(batch_size).view(-1, 1).cuda()
        print('lossG trueTensor', trueTensor.size())
        print('fakeSource', fakeSource.size())
        lossG = criterionSource(fakeSource, trueTensor.expand_as(fakeSource))
        optimizerG.zero_grad()
        optimizerA.zero_grad()
        lossG.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(),20)
        torch.nn.utils.clip_grad_norm_(A.parameters(),20)
        optimizerG.step()
        optimizerA.step()
        
        board.add_scalar('realLoss', realLoss.item(), step)
        board.add_scalar('fakeLoss', fakeLoss.item(), step)
        board.add_scalar('lossD', lossD.item(), step)
        board.add_scalar('lossG', lossG.item(), step)
    if (epoch + 1) % 50 == 0:
        torch.save(G.state_dict(), DIRNAME + "G" + str(epoch) + ".pt")
        torch.save(D.state_dict(), DIRNAME + "D" + str(epoch) + ".pt")
    if (epoch + 1) % 10 == 0:   
        with torch.no_grad():
            G.eval()
            sample_image(epoch)
            G.train()
