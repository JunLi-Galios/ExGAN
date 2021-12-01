import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class Extremeness(Object):
    def __init__(self, mu):
        super(Extremeness，self).__init__()
        self.mu = mu
    
    def cal_extreme(self, inp):
        raise NotImplementedError
        
    def func(self):
        raise NotImplementedError
        
    def optimize(self, inp_size):        
        X = torch.nn.Parameter(torch.randn(inp_size) * 0.001)
        fn = self.func()
        loss = - fn(X)
        optimizer = optim.SGD(X, lr=0.001)
        while loss > -self.mu:
            optimizer.zero_grad()
            lossG.backward()
            optimizer.step()
            loss = - fn(X)
            
        return X.data
        
        
class AvgExtremeness(Extremeness):
    def __init__(self, mu):
        super(AvgExtremeness，self).__init__(self, mu)
        print('AvgExtremeness mu', mu)
        
    def cal_extreme(self, inp):
        batch = inp.size()[0]
        re_inp = inp.reshape(batch, -1)
        return torch.mean(re_inp, dim=1)
    
    def func(self):
        fn = lambda x: torch.mean(x)
        return fn
    
    
    
class MaxExtremeness(Extremeness):
    def __init__(self,mu):
        super(MaxExtremeness，self).__init__(self, mu)
        print('MaxExtremeness mu', mu)
        
    def cal_extreme(inp):
        batch = inp.size()[0]
        re_inp = inp.reshape(batch, -1)
        return torch.max(re_inp, dim=1).values
    
    def func(self):
        fn = lambda x: torch.max(x)
        return fn
    
    

    
    
