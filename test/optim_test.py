"""
test for nn/optim/PGManager.py,NFI.py
"""
#%%
from numpy import *
import numpy as np
import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from aTEAM.optim import ParamGroupsManager,NumpyFunctionInterface
#%%
device0 = -1
device1 = -1
device2 = -1
def generate_wb(device):
    if device<0:
        return [torch.FloatTensor(3,3).normal_(),torch.FloatTensor(3).normal_()]
    else:
        w,b = generate_wb(-1)
        return w.cuda(device),b.cuda(device)
weight0,bias0 = generate_wb(device0)
weight1,bias1 = generate_wb(device1)
weight2,bias2 = generate_wb(device2)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.weight0 = nn.Parameter(weight0)
        self.bias0 = nn.Parameter(bias0)
        self.weight1 = nn.Parameter(weight1)
        self.bias1 = nn.Parameter(bias1)
        self.weight2 = nn.Parameter(weight2)
        self.bias2 = nn.Parameter(bias2)
    def forward(self, input):
        p_w0 = self.weight0.cpu()
        p_b0 = self.bias0.cpu()
        p_w1 = self.weight1.cpu()
        p_b1 = self.bias1.cpu()
        p_w2 = self.weight2.cpu()
        p_b2 = self.bias2.cpu()
        input = input.cpu()
        out = F.softmax(p_w0@input+p_b0, dim=0)*\
                F.softmax(p_w1@input+p_b1, dim=0)*\
                F.softmax(p_w2@input+p_b2, dim=0)
        out = out.sum()
        return out
net = Net()
def forward_gen():
    I = torch.FloatTensor(3).normal_()
    def forward():
        return net(I)
    return forward
def x_proj(param_group):
    param_group['w'].data[0,1] = 1
    param_group['b'].data[0] = 1
def grad_proj(param_group):
    if not param_group['w'].grad is None:
        param_group['w'].grad.data[0,1] = 0
    if not param_group['b'].grad is None:
        param_group['b'].grad.data[0] = 0
forward = forward_gen()
param_group0 = {'params':{'w':net.weight0,'b':net.bias0}}
param_group1 = {'params':{'w':net.weight1,'b':net.bias1}}
param_group2 = {'params':iter([net.weight2,net.bias2])}
nfi = NumpyFunctionInterface([param_group0,param_group1], forward=forward)
a0 = random.randn(1000).astype(dtype=np.float32)
def test():
    a = a0[:nfi.numel()].copy()
    f = nfi.f(a)
    g = nfi.fprime(a)
    print(g)
    print(a-nfi.flat_param)
    print(nfi.f(a)-f)
    print(np.linalg.norm(nfi.fprime(a)-g))
    nfi.fprime(a+1)
    print(nfi.f(a)-f)
    print(nfi.flat_param-a)
    nfi.flat_param = a
    print(nfi.fprime(a)-g)
print('no frozen, no x_proj, no grad_proj, 2 param_groups')
test()
nfi.set_options(1,x_proj=x_proj,grad_proj=grad_proj)
print('set x_proj and grad_proj in param_group[1], 2 param_groups')
test()
nfi.add_param_group(param_group2)
print('add param_group')
test()
nfi.flat_param = random.randn(nfi.numel())
nfi.set_options(1,isfrozen=True)
print('set frozen in param_group[1], 3 param_groups')
test()
nfi.set_options(1,x_proj=None,grad_proj=None)
print('delete x_proj,grad_proj in param_group[1], 3 param_groups')
test()
nfi.set_options(0,x_proj=x_proj,grad_proj=grad_proj)
print('add x_proj and grad_proj in param_group[0], 3 param_groups')
test()
nfi.forward = forward_gen()
print('change forward function')
test()
nfi.flat_param = a0[:nfi.numel()]+10
print('change flat_param')
print(nfi.forward()-nfi.f(a0[:nfi.numel()]+10))
test()
nfi.set_options(1,isfrozen=False)
print('set free in param_group[1], 3 param_groups')
test()
#%%

