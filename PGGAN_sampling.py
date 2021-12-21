from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import time
from scipy.stats import genpareto
import torch.nn.functional as F
from torch.autograd import Variable
from torch import FloatTensor

import argparse
parser = argparse.ArgumentParser(description='PGGAN_sampling')
parser.add_argument('--save', default='', type=str,
                    help='save parameters and logs in this folder')

# Dataset options
# parser.add_argument('--dataset', default='real', type=str)

# Model options
parser.add_argument('--model', default='finetune', type=str)
parser.add_argument('--simple', action='store_true', default=False)

args = parser.parse_args()

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
        self.block3 = convBNReLU(128, 256)
        self.block4 = convBNReLU(256, 512)
        self.block5 = nn.Conv2d(512, 64, 4, 1, 0)
        out_channels = np.prod(img_size)
        self.mu = nn.Linear(64, out_channels)
        self.sigma = nn.Linear(64, out_channels)
        self.gamma = nn.Linear(64, out_channels)
        self.img_size = img_size

    def forward(self, inp):
        out = self.block1(inp) 
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        size = out.shape[0]
#         print('out', out.size())
        out = out.view(size, -1)
#         print('out', out.size())
        mu = torch.abs(self.mu(out))
        sigma = torch.abs(self.sigma(out))
        gamma = torch.abs(self.gamma(out))
        return torch.reshape(mu, [size] + self.img_size), torch.reshape(sigma, [size] + self.img_size), \
            torch.reshape(gamma, [size] + self.img_size)
        
    
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.block1 = nn.Conv2d(1, 4, 3, 1, 1)
        self.block2 = nn.Conv2d(4, 4, 3, 1, 1)
        self.block3 = nn.Conv2d(4, 4, 3, 1, 1)
        self.block4 = nn.Conv2d(4, 1, 3, 1, 1)

    def forward(self, inp):
        out = self.block1(inp) 
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        return out

latentdim = 20
G = Generator(in_channels=latentdim, out_channels=1).cuda()
G.load_state_dict(torch.load('{}/G999.pt'.format(args.save)))
G.eval()
if args.model == 'finetune':
    T = Transformer().cuda()
else:
    T = nn.Identity().cuda()
T.eval()
mu = torch.load('{}/mu999.pt'.format(args.save)).cuda()
sigma = torch.load('{}/sigma999.pt'.format(args.save)).cuda()
gamma = torch.load('{}/gamma999.pt'.format(args.save)).cuda()
e = torch.distributions.exponential.Exponential(torch.ones([1] + img_size))

t = time.time()
latent = Variable(FloatTensor(torch.randn(100, latentdim, 1, 1))).cuda()
fakeData = G(latent)
max_value, _ = torch.max(torch.reshape(fakeData, [100, -1]), dim=1)
max_value = torch.reshape(max_value, [-1, 1, 1, 1])
G_samples = fakeData - max_value
e_samples = e.rsample([len(G_samples)]).cuda()
G_extremes = sigma * (G_samples + e_samples) + mu
print(time.time() - t)
torch.save(0.5*(G_extremes+1), '{}/PxGAN_sample.pt'.format(args.save))
