"""
2d examples for LagrangeInterp,LagrangeInterpFixInputs (nn.modules.Interpolation)
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
from aTEAM.nn.modules import LagrangeInterp,LagrangeInterpFixInputs
from aTEAM.utils import meshgen
#%%
def testfunc(inputs):
    """inputs (ndarray)"""
    return sin(inputs[...,0]*1.8*pi)-cos(inputs[...,1]*2.3*pi)
def compare(I, inputs):
    infe = I(inputs).data.cpu().numpy()
    infe_true = testfunc(inputs.data.cpu().numpy())
    return infe,infe_true
def forward(I, inputs):
    outputs = I(inputs)
    outputs_true = torch.from_numpy(testfunc(inputs.data.cpu().numpy()))
    outputs_true = outputs.data.new(outputs_true.size()).copy_(outputs_true)
    return ((outputs-outputs_true)**2).mean()
def forwardFixInputs(IFixInputs, outputs_true):
    outputs = IFixInputs()
    return ((outputs-outputs_true)**2).mean()
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
I = LagrangeInterp(m,d,mesh_bound,mesh_size)
I.double()
if device>=0:
    I.cuda(device)
mesh_bound[1] += 1/1000
dataset = meshgen(mesh_bound, [1001,1001])
dataset = torch.from_numpy(dataset)
dataset = I.interp_coe.data.new(dataset.size()).copy_(dataset)
mesh_bound[1] -= 1/1000
IFixInputs = LagrangeInterpFixInputs(dataset,m,d,mesh_bound,mesh_size)
IFixInputs.double()
if device>=0:
    IFixInputs.cuda(device)
ax = plt.figure().add_subplot(1,1,1)
ax.imshow(I(dataset).data.cpu().numpy())
#%%
nfi = NumpyFunctionInterface([I.interp_coe,],forward=lambda :forward(I,dataset))
nfi.flat_param = random.randn(nfi.numel())
x,f,d = lbfgsb(nfi.f,nfi.flat_param,nfi.fprime,m=1000,factr=1,pgtol=1e-14,iprint=10)
infe,infe_true = compare(I, dataset)
ax = plt.figure().add_subplot(1,1,1)
ax.imshow(infe)
errs = infe-infe_true
ax = plt.figure().add_subplot(1,1,1)
ax.imshow(errs)
#%%
outputs = IFixInputs()
outputs_true = torch.from_numpy(testfunc(IFixInputs.inputs.cpu().numpy()))
outputs_true = outputs_true.view(outputs.size())
outputs_true = outputs.data.new(outputs_true.size()).copy_(outputs_true)
nfi = NumpyFunctionInterface(IFixInputs.parameters(), forward=lambda :forwardFixInputs(IFixInputs, outputs_true))
nfi.flat_param = random.randn(nfi.numel())
x,f,d = lbfgsb(nfi.f,nfi.flat_param,nfi.fprime,m=1000,factr=1,pgtol=1e-14,iprint=10)
infe = IFixInputs().data.cpu().numpy()
infe_true = outputs_true.data.cpu().numpy()
ax = plt.figure().add_subplot(1,1,1)
ax.imshow(infe)
errs = infe-infe_true
ax = plt.figure().add_subplot(1,1,1)
ax.imshow(errs)
#%%
inputs = dataset[5::10,5::10].clone()
print(sqrt(forward(I,inputs).data.item()))
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


