"""
test for nn/modules/MK.py
"""
#%%
from numpy import *
import numpy as np
import torch
from torch.autograd import grad
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import aTEAM.nn
from aTEAM.nn.modules import M2K,K2M
from aTEAM.utils import switch_moment_filter, diff_monomial_coe
#%%
shape = [7,3,5]
m2k = M2K(shape)
k2m = K2M(shape)
a = torch.randn(*([10]+shape)).double()
npm2k,npk2m,_,_ = switch_moment_filter(shape)
m2ka = m2k(a).data.cpu().numpy()
k2ma = k2m(a).data.cpu().numpy()
packa = m2k._packdim(a).data.cpu().numpy()
npm2ka = []
npk2ma = []
for i in range(packa.shape[0]):
    npm2ka.append(npm2k(packa[i]))
    npk2ma.append(npk2m(packa[i]))
npm2ka = reshape(stack(npm2ka), list(a.size()))
npk2ma = reshape(stack(npk2ma), list(a.size()))
print(linalg.norm(npm2ka-m2ka))
print(linalg.norm(npk2ma-k2ma))
#%%
shape = [5,7]
ker = diff_monomial_coe(x_order=2,y_order=1,shape=shape)
a = torch.from_numpy(ker).double()
m2k = M2K(shape)
k2m = K2M(shape)
npm2k,npk2m,_,_ = switch_moment_filter(shape)
m2ka = m2k(a).data.cpu().numpy()
k2ma = k2m(a).data.cpu().numpy()
packa = m2k._packdim(a).data.cpu().numpy()
npm2ka = []
npk2ma = []
for i in range(packa.shape[0]):
    npm2ka.append(npm2k(packa[i]))
    npk2ma.append(npk2m(packa[i]))
npm2ka = reshape(stack(npm2ka), list(a.size()))
npk2ma = reshape(stack(npk2ma), list(a.size()))
print(linalg.norm(npm2ka-m2ka))
print(linalg.norm(npk2ma-k2ma))
#%%

