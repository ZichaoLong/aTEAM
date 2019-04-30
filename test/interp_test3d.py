"""
3d example for lagrangeinterp (nn.functional.interpolation)
"""
#%%
from numpy import *
import numpy as np
import scipy.sparse
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
    return sin(inputs[...,0]*8)+cos(sqrt(inputs[...,1]*4))*sin(inputs[...,2]*4)

class Interp(nn.Module):
    def __init__(self, m, d, mesh_bound, mesh_size, device=-1):
        super(Interp,self).__init__()
        self.m = m
        self.d = d
        self.mesh_bound = mesh_bound.copy()
        self.mesh_size = mesh_size.copy()
        mesh_size = list(map(lambda x:int(x), list(mesh_size*d+1)))
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
m = 3
d = 2
device = -1
mesh_bound = zeros((2,m))
# mesh_bound[0] = arange(m)-1
# mesh_bound[1] = arange(m)+1
mesh_bound[0] = 0
mesh_bound[1] = 1
mesh_size = array([40,]*m)
I = Interp(m,d,mesh_bound, mesh_size, device=device)
mesh_bound[1] += 1/200
dataset = meshgen(mesh_bound, [201,201,201])
dataset = torch.from_numpy(dataset).clone()
dataset = I.interp_coe.data.new(dataset.size()).copy_(dataset)
nfi = NumpyFunctionInterface([I.interp_coe,],forward=lambda :I.forward(dataset))
nfi.flat_param = random.randn(nfi.numel())
x0 = nfi.flat_param
#%%
inputs_shape = [50,50,50]
IN,JN,KN = int(200/inputs_shape[0]), int(200/inputs_shape[1]), int(200/inputs_shape[2])
indx = zeros((IN*JN*KN,3),dtype=int32)
idx = 0
for i in range(IN):
    for j in range(JN):
        for k in range(KN):
            indx[idx] = array([i,j,k])*array(inputs_shape)
            idx += 1
for i in range(64):
    inputs = dataset[
            indx[i,0]:indx[i,0]+inputs_shape[0],
            indx[i,1]:indx[i,1]+inputs_shape[1],
            indx[i,2]:indx[i,2]+inputs_shape[2]
            ]
    inputs = inputs.clone()
    nfi.forward = lambda :I.forward(inputs)
    x = nfi.flat_param
    x,f,d = lbfgsb(nfi.f,x,nfi.fprime,m=1000,maxiter=20,factr=1,pgtol=1e-16,iprint=10)
#%%
inputs = dataset[
        random.randint(200/inputs_shape[0])+int(200/inputs_shape[0])*arange(0,inputs_shape[0],dtype=int32)[:,newaxis,newaxis],
        random.randint(200/inputs_shape[1])+int(200/inputs_shape[1])*arange(0,inputs_shape[1],dtype=int32)[newaxis,:,newaxis],
        random.randint(200/inputs_shape[2])+int(200/inputs_shape[2])*arange(0,inputs_shape[2],dtype=int32)[newaxis,newaxis,:]
        ]
inputs = inputs.clone()
nfi.forward = lambda :I.forward(inputs)
infe,infe_true = compare(I,inputs)
print(sqrt((infe-infe_true)**2).mean())
print(sqrt((infe-infe_true)**2).max())
h = plt.figure()
indx = random.randint(20)
a = h.add_subplot(4,2,1)
a.imshow(infe_true[indx])
a.set_title('true')
a = h.add_subplot(4,2,2)
a.imshow(infe[indx])
a.set_title('inferenced')
indx = random.randint(20)
a = h.add_subplot(4,2,3)
a.plot(infe_true[indx,indx])
a = h.add_subplot(4,2,4)
a.plot(infe[indx,indx])
indx = random.randint(20)
a = h.add_subplot(4,2,5)
a.plot(infe_true[indx,:,indx])
a = h.add_subplot(4,2,6)
a.plot(infe[indx,:,indx])
indx = random.randint(20)
a = h.add_subplot(4,2,7)
a.plot(infe_true[:,indx,indx])
a = h.add_subplot(4,2,8)
a.plot(infe[:,indx,indx])
#%%
