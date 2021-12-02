import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.optim as optim

class Extremeness:
    def __init__(self):
        super(Extremeness, self).__init__()
    
    def cal_extreme(self, inp):
        raise NotImplementedError
        
    def func(self):
        raise NotImplementedError
        
    def init_func(self, inp_size, mu):
        raise NotImplementedError
        
    def optimize(self, inp_size, mu):        
        X = torch.nn.Parameter(self.init_func(inp_size, mu))
        mu = mu.cuda()
        fn = self.func()
        loss = - fn(X).cuda()
        optimizer = optim.SGD([X], lr=0.001)
        while loss > -mu:
            optimizer.zero_grad()
            lossG.backward()
            optimizer.step()
            loss = - fn(X)
            
        return X.data
        
        
class AvgExtremeness(Extremeness):
    def __init__(self):
        super(AvgExtremeness, self).__init__()
        
    def cal_extreme(self, inp):
        batch = inp.size()[0]
        re_inp = inp.reshape(batch, -1)
        return torch.mean(re_inp, dim=1)
    
    def func(self):
        fn = lambda x: torch.mean(x)
        return fn
    
    def init_func(self, inp_size, mu):
        raise  torch.ones(inp_size) * mu
    
    
    
class MaxExtremeness(Extremeness):
    def __init__(self):
        super(MaxExtremeness, self).__init__()
        
    def cal_extreme(self, inp):
        batch = inp.size()[0]
        re_inp = inp.reshape(batch, -1)
        return torch.max(re_inp, dim=1).values
    
    def func(self):
        fn = lambda x: torch.max(x)
        return fn
   

