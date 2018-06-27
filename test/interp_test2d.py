"""
2d example for lagrangeinterp (nn.functional.interpolation)
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
import matplotlib.pyplot as plt
from aTEAM.optim import NumpyFunctionInterface,ParamGroupsManager
from aTEAM.nn.functional import lagrangeinterp
from aTEAM.utils import meshgen
#%%
def testfunc(inputs):
    """inputs (ndarray)"""
    return sin(inputs[...,0]*1.8*pi)-cos(inputs[...,1]*2.3*pi)

class Interp(nn.Module):
    def __init__(self, m, d, mesh_bound, mesh_size, device=-1):
        super(Interp,self).__init__()
        self.m = m
        self.d = d
        self.mesh_bound = mesh_bound.copy()
        self.mesh_size = mesh_size.copy()
        mesh_size = list(map(lambda x:int(x), list(mesh_size*max(d,1)+1)))
        interp_coe = torch.DoubleTensor(*mesh_size).normal_()
        if device>=0:
            interp_coe = interp_coe.cuda(device)
        self.interp_coe = nn.Parameter(interp_coe)
    def infe(self, inputs):
        if not inputs.is_contiguous():
            inputs = inputs.clone()
        inputs_size = inputs.size()
        outputs = lagrangeinterp(inputs.view([-1,self.m]), interp_coe=self.interp_coe, \
                interp_dim=self.m, interp_degree=self.d, \
                mesh_bound=self.mesh_bound, \
                mesh_size=self.mesh_size).view(inputs_size[:-1])
        return outputs
    def forward(self, inputs):
        outputs = self.infe(inputs)
        outputs_true = torch.from_numpy(testfunc(inputs.data.cpu().numpy()))
        outputs_true = outputs.data.new(outputs_true.size()).copy_(outputs_true)
        return ((outputs-outputs_true)**2).mean()
def compare(I, inputs):
    infe = I.infe(inputs).data.cpu().numpy()
    infe_true = testfunc(inputs.data.cpu().numpy())
    return infe,infe_true
#%%
m = 2
d = 2
device = -1
mesh_bound = zeros((2,m))
# mesh_bound[0] = arange(m)-1
# mesh_bound[1] = arange(m)+1
mesh_bound[0] = 0
mesh_bound[1] = 1
mesh_size = array([40,]*m)
I = Interp(m,d,mesh_bound, mesh_size, device=device)
mesh_bound[1] += 1/1000
dataset = meshgen(mesh_bound, [1001,1001])
dataset = torch.from_numpy(dataset)
dataset = I.interp_coe.data.new(dataset.size()).copy_(dataset)
ax = plt.figure().add_subplot(1,1,1)
ax.imshow(I.infe(dataset).data.cpu().numpy())
#%%
nfi = NumpyFunctionInterface([I.interp_coe,],forward=lambda :I.forward(dataset))
nfi.flat_param = random.randn(nfi.numel())
x,f,d = lbfgsb(nfi.f,nfi.flat_param,nfi.fprime,m=1000,factr=1,pgtol=1e-14,iprint=10)
infe,infe_true = compare(I, dataset)
ax = plt.figure().add_subplot(1,1,1)
ax.imshow(infe)
errs = infe-infe_true
ax = plt.figure().add_subplot(1,1,1)
ax.imshow(errs)
#%%
# nfi.flat_param = random.randn(nfi.numel())
for iii in range(50):
    inputs = dataset[random.choice(1000,100)[:,newaxis],random.choice(1000,100)[newaxis,:]]
    nfi.forward = lambda :I.forward(inputs)
    x,f,d = lbfgsb(nfi.f,nfi.flat_param,nfi.fprime,m=1000,maxiter=10,factr=1,pgtol=1e-14,iprint=10)
#%%
inputs = dataset[5::10,5::10].clone()
print(sqrt(I.forward(inputs).item()))
infe,infe_true = compare(I,inputs)
plt.figure().add_subplot(111).imshow(infe-infe_true)
h = plt.figure()
a = h.add_subplot(3,2,1)
a.imshow(infe_true)
a.set_title('true')
a = h.add_subplot(3,2,2)
a.imshow(infe)
a.set_title('inferenced')
indx = random.randint(100)
a = h.add_subplot(3,2,3)
a.plot(infe_true[indx])
a = h.add_subplot(3,2,4)
a.plot(infe[indx])
indx = random.randint(100)
a = h.add_subplot(3,2,5)
a.plot(infe_true[:,indx])
a = h.add_subplot(3,2,6)
a.plot(infe[:,indx])

#%%
