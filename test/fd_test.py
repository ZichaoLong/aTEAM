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
from scipy.signal import correlate,correlate2d
import matplotlib.pyplot as plt
import aTEAM.nn
from aTEAM.nn.modules import M2K,K2M
from aTEAM.nn.modules import FD1d,FD2d,FD3d
from aTEAM.utils import switch_moment_filter, diff_monomial_coe
#%% 1d
f1 = FD1d(7,2,constraint='moment',boundary='Dirichlet')
inputs = torch.randn(1,10,dtype=torch.float64)
f1.kernel = random.randn(*list(f1.kernel.shape))
outputs = f1(inputs)
outputs_np = correlate(inputs[0].detach().numpy(),f1.kernel.detach().numpy(),mode='same')
print(linalg.norm(outputs.detach().numpy()-outputs_np))
f1.x_proj()
print(f1.moment.data)
print(f1.kernel)
outputs = f1(inputs)
outputs_np = correlate(inputs[0].detach().numpy(),f1.kernel.detach().numpy(),mode='same')
print(linalg.norm(outputs.detach().numpy()-outputs_np))
f1.constraint = 2
f1.x_proj()
print(f1.moment.data)
print(f1.kernel)
outputs = f1(inputs)
outputs_np = correlate(inputs[0].detach().numpy(),f1.kernel.detach().numpy(),mode='same')
print(linalg.norm(outputs.detach().numpy()-outputs_np))
#%% 2d
f2 = FD2d((5,3),(1,0),constraint='moment',boundary='Dirichlet')
inputs = torch.randn(1,10,10,dtype=torch.float64)
f2.kernel = random.randn(*list(f2.kernel.shape))
outputs = f2(inputs)
outputs_np = correlate(inputs[0].detach().numpy(),f2.kernel.detach().numpy(),mode='same')
f3 = FD3d(7,(1,0,0),constraint='moment',boundary='Dirichlet')
f2.x_proj()
print(f2.moment.data)
print(f2.kernel)
outputs = f2(inputs)
outputs_np = correlate(inputs[0].detach().numpy(),f2.kernel.detach().numpy(),mode='same')
print(linalg.norm(outputs.detach().numpy()-outputs_np))
f2.constraint = 2
f2.x_proj()
print(f2.moment.data)
print(f2.kernel)
outputs = f2(inputs)
outputs_np = correlate(inputs[0].detach().numpy(),f2.kernel.detach().numpy(),mode='same')
print(linalg.norm(outputs.detach().numpy()-outputs_np))
#%% 3d
f3 = FD3d(3,(1,0,0),constraint='moment',boundary='Dirichlet')
inputs = torch.randn(1,10,10,10,dtype=torch.float64)
f3.kernel = random.randn(*list(f3.kernel.shape))
outputs = f3(inputs)
outputs_np = correlate(inputs[0].detach().numpy(),f3.kernel.detach().numpy(),mode='same')
f3.x_proj()
print(f3.moment.data)
print(f3.kernel)
outputs = f3(inputs)
outputs_np = correlate(inputs[0].detach().numpy(),f3.kernel.detach().numpy(),mode='same')
print(linalg.norm(outputs.detach().numpy()-outputs_np))
f3.constraint = 2
f3.x_proj()
print(f3.moment.data)
print(f3.kernel)
outputs = f3(inputs)
outputs_np = correlate(inputs[0].detach().numpy(),f3.kernel.detach().numpy(),mode='same')
print(linalg.norm(outputs.detach().numpy()-outputs_np))
#%%
