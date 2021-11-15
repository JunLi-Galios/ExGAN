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

class NWSDataset(Dataset):
    """
    NWS Dataset
    """

    def __init__(
        self, path='data/', dsize=2557
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
    def __init__(self, in_channels, img_size):
        super(Aggregator, self).__init__()
        self.in_channels = in_channels
        self.block1 = convBNReLU(self.in_channels, 64)
        self.block2 = convBNReLU(64, 128)
        self.block5 = nn.Conv2d(128, 64, 4, 1, 0)
        out_channels = 2 * np.prod(img_size)
        self.source = nn.Linear(64, out_channels)
        self.img_size = img_size + [2]

    def forward(self, inp):
        out = self.block1(inp) 
        out = self.block2(out)
#         out = self.block3(out)
#         out = self.block4(out)
        out = self.block5(out)
        size = out.shape[0]
        out = out.view(size, -1)
        source = torch.abs(self.source(out))
        source = torch.reshape(source, self.img_size)
        return source

latentdim = 20
img_size = (64, 64)
criterionSource = nn.BCELoss()
criterionContinuous = nn.L1Loss()
criterionValG = nn.L1Loss()
criterionValD = nn.L1Loss()
G = Generator(in_channels=latentdim, out_channels=1).cuda()
D = Discriminator(in_channels=1).cuda()
A = Aggregator(img_size).cuda()
G.apply(weights_init_normal)
D.apply(weights_init_normal)
A.apply(weights_init_normal)

optimizerG = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizerA = optim.Adam(A.parameters(), lr=0.0001, betas=(0.5, 0.999))
static_z = Variable(FloatTensor(torch.randn((81, latentdim, 1, 1)))).cuda()

def sample_image(batches_done):
    static_sample = G(static_z).detach().cpu()
    static_sample = (static_sample + 1) / 2.0
    save_image(static_sample, DIRNAME + "%d.png" % batches_done, nrow=9)

DIRNAME = 'PGGAN/'
os.makedirs(DIRNAME, exist_ok=True)

board = SummaryWriter(log_dir=DIRNAME)

def pick_samples(samples, u):
    flag_list = []
    print(samples.size())
    for i in range(len(u)):
        flag = samples[:,i] > u[i]
        flag_list.append(flag)
        print(flag)

    flags = torch.stack(flag_list, dim=1)
    print('flags', flags.size())
    
    flags.max()
    total_flag = samples[:,0] < -np.inf
    
    for i in range(len(u)):
        total_flag = total_flag | flag_list[i]
        
    print(total_flag)
    return samples[total_flag,:]

step = 0
ratio = 0.001
# mu = torch.ones(img_size)
para = torch.ones(img_size + [2])
e = torch.distributions.exponential.Exponential(torch.ones(img_size))
n_extremes_list = []
acc_list = []


for epoch in range(1000):
    print(epoch)
    for images in dataloader:
        para_val = A(images)
        print('para_val size', para_val.size())
        para_incre = torch.mean(torch.abs(para_val),dim=0)
        para = (1 - ratio) * para + ratio * para_incre
        mu = para[:,:,0]
        sigma = para[:,:,1]
        
        extreme_samples = pick_samples(samples, u) - u
        n_extremes = len(extreme_samples)
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
        print('realSource', realSource.size())
        realSource = D(extreme_samples + noise*torch.randn_like(extreme_samples).cuda())
        realLoss = criterionSource(realSource, trueTensor.expand_as(realSource))
        
        
        latent = Variable(torch.randn(n_extremes, latentdim, 1, 1)).cuda()
        
        fakeData = G(latent)
        print('fakeData', fakeData.size())
        max_value, _ = torch.max(fakeData, dim=0)
        G_samples = fakeData - max_value
        e_samples = e.rsample([len(G_samples)])
        G_extremes = sigma * (G_samples + e_samples)
        
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

