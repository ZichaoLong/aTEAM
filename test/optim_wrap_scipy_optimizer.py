"""
test for nn/optim/PGManager.py,NFI.py
"""
#%%
from numpy import *
import numpy as np
import torch
from torch.autograd import grad
import torch.nn as nn
from torch.nn import functional as F
from scipy.optimize.lbfgsb import fmin_l_bfgs_b as lbfgsb
from scipy.optimize.slsqp import fmin_slsqp as slsqp
from aTEAM.optim import NumpyFunctionInterface,ParamGroupsManager
#%%
device1 = -1
device2 = -1
class Penalty(nn.Module):
    def __init__(self,n,alpha=1e-5):
        super(Penalty,self).__init__()
        m = n//2
        x1 = torch.arange(1,m+1).to(torch.float64)
        x2 = torch.arange(m+1,n+1).to(torch.float64)
        if device1>=0:
            x1 = x1.cuda(device1)
        if device2>=0:
            x2 = x2.cuda(device2)
        self.x1 = nn.Parameter(x1)
        self.x2 = nn.Parameter(x2)
        self.n = n
        self.alpha = alpha
    def forward(self):
        x = torch.cat([self.x1.cpu(),self.x2.cpu()],0)
        return self.alpha*((x-1)**2).sum()+((x**2).sum()-0.25)**2
class Trignometric(nn.Module):
    def __init__(self,n):
        super(Trignometric,self).__init__()
        self.x = nn.Parameter(torch.zeros(n, dtype=torch.float64).fill_(1/n))
        self.n = n
    def forward(self):
        n = self.n
        x = self.x
        y = x.cos()
        z = x.sin()
        s = n-y.sum()
        return ((s+torch.arange(1,n+1).to(torch.float64)*(1-y)-z)**2).sum()
def biggs_exp6(x,m=6):
    t = torch.arange(1,m+1).to(torch.float64)
    y = (-t).exp()-5*(-10*t).exp()+3*(-4*t).exp()
    te1 = (-x[0]*t).exp()
    te2 = (-x[1]*t).exp()
    te3 = (-x[4]*t).exp()
    r = x[2]*te1-x[3]*te2+x[5]*te3-y
    return r.dot(r)
def powell_bs(x):
    return (1e4*x[0]*x[1]-1)**2+((-x[0]).exp()+(-x[1]).exp()-1.0001)**2
#%% Penalty
penalty = Penalty(100,1e-5)
nfi = NumpyFunctionInterface(
        [{'params':[penalty.x1,]},
            {'params':[penalty.x2,]}],
        penalty.forward)
x0 = torch.cat([penalty.x1.cpu(),penalty.x2.cpu()],0).data.clone().numpy()
x,f,d = lbfgsb(nfi.f,x0,nfi.fprime,m=100,factr=1,pgtol=1e-14,iprint=10)
out,fx,its,imode,smode = slsqp(nfi.f,x0,fprime=nfi.fprime,acc=1e-16,iter=15000,iprint=1,full_output=True)
#%% Trignometric
trig = Trignometric(100)
nfi = NumpyFunctionInterface(trig.parameters(),trig.forward)
x0 = trig.x.data.clone().numpy()
x,f,d = lbfgsb(nfi.f,x0,nfi.fprime,m=100,factr=1,pgtol=1e-14,iprint=10)
out,fx,its,imode,smode = slsqp(nfi.f,x0,fprime=nfi.fprime,acc=1e-16,iter=15000,iprint=1,full_output=True)
#%%
class BIGGS_EXP(object):
    def __init__(self,x,m):
        self.x = x
        self.m = m
    def __call__(self):
        return biggs_exp6(self.x,self.m)
forward = BIGGS_EXP(torch.tensor([1,2,1,1,1,1],dtype=torch.float64,requires_grad=True),6)
nfi = NumpyFunctionInterface([forward.x,],forward=forward)
x0 = forward.x.data.clone().numpy()
xopt = array([1,10,1,5,4,3])
x1 = xopt+1e-2*random.randn(6)
x,f,d = lbfgsb(nfi.f,x0,nfi.fprime,m=100,factr=1,pgtol=1e-14,iprint=10)
out,fx,its,imode,smode = slsqp(nfi.f,x0,fprime=nfi.fprime,acc=1e-16,iter=15000,iprint=1,full_output=True)
nfi.forward.m = 13
x,f,d = lbfgsb(nfi.f,x0,nfi.fprime,m=100,factr=1,pgtol=1e-14,iprint=10)
out,fx,its,imode,smode = slsqp(nfi.f,x0,fprime=nfi.fprime,acc=1e-16,iter=15000,iprint=1,full_output=True)
#%%
nfix = torch.tensor([0,1], dtype=torch.float64, requires_grad=True)
def forward():
    return powell_bs(nfix)
nfi = NumpyFunctionInterface([nfix,],forward=forward)
x0 = array([0,1])
x,f,d = lbfgsb(nfi.f,x0,nfi.fprime,m=100,factr=1,pgtol=1e-14,iprint=10)
out,fx,its,imode,smode = slsqp(nfi.f,x0,fprime=nfi.fprime,acc=1e-16,iter=15000,iprint=1,full_output=True)
def x_proj(params):
    params[0].data[0] = 1e-5
def grad_proj(params):
    params[0].grad.data[0] = 0
nfi.set_options(0,x_proj=x_proj,grad_proj=grad_proj)
x,f,d = lbfgsb(nfi.f,x0,nfi.fprime,m=100,factr=1,pgtol=1e-14,iprint=10)
x = nfi.flat_param
out,fx,its,imode,smode = slsqp(nfi.f,x0,fprime=nfi.fprime,acc=1e-16,iter=15000,iprint=1,full_output=True)
out = nfi.flat_param
#%%

