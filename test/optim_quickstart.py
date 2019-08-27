"""
A quick start for aTEAM/optim
More test example can be found in aTEAM/test/optim*.py. 
"""
from numpy import *
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from aTEAM.optim import NumpyFunctionInterface,ParamGroupsManager
"""
Example 1
Let us start with an example.
At first, we define a PyTorch tensor function "powell_bs"  
"""
def powell_bs(x):
    return (1e4*x[0]*x[1]-1)**2+((-x[0]).exp()+(-x[1]).exp()-1.0001)**2
"""
And then define the variable "nfix" to be optimized:
    min_{nfix} powell_bs(nfix)
"""
nfix = torch.tensor([0,1], dtype=torch.float64, requires_grad=True)
"""
a interface "forward" for NumpyFunctionInterface is needed
"""
def forward():
    return powell_bs(nfix)
"""
At last, construct your NumpyFunctionInterface of the PyTorch tensor function
"""
listofparameters = [nfix,]
nfi = NumpyFunctionInterface(listofparameters,forward=forward)
"""
Now it's ready to use interfaces given by "nfi": "nfi.flat_param,nfi.f,nfi.fprime". 
What these interfaces do is somethine like
```
class NumpyFunctionInterface:
    @property
    def params(self):
    # notice that nfi = NumpyFunctionInterface(listofparameters,forward)
        for p in listofparameters:
            yield p
    @property
    def flat_param(self):
        views = []
        for p in self.params:
            views.append(p.view(-1))
        return torch.cat(views,0).numpy()
    @property.setter
    def flat_param(self,x): # x is a numpy array
        for p in self.params: 
            p[:] = x[pidx_start:pidx_end] 
            # For simplicity here we do not show details of 
            # type conversion and subscript matching between p and x.
    def f(self,x): 
        self.flat_param = x
        return forward()
    def fprime(self,x):
        loss = self.f(x)
        loss.backward() # Here we utilize autograd feature of PyTorch
        grad = np.zeros(x.size)
        for p in self.params:
            grad[pidx_start:pidx_end] = p.grad
        return grad
```
Try these commands:
    x = np.random.randn(nfi.numel())
    assert(np.equal(nfi.f(x),powell_bs(x)))
    x[0] = 1
    nfi.flat_param = x 
    # nfi.flat_param[0] = 1 is not permitted since property is not a ndarray
    assert(np.equal(nfi.f(nfi.flat_param),powell_bs(x)))
These interfaces enable us to use lbfgs,slsqp from scipy.optimize.
"""
from scipy.optimize.lbfgsb import fmin_l_bfgs_b as lbfgsb
from scipy.optimize.slsqp import fmin_slsqp as slsqp
x0 = array([0,1])
print(" ***************** powell_bs ***************** ")
x,f,d = lbfgsb(nfi.f,x0,nfi.fprime,m=100,factr=1,pgtol=1e-14,iprint=10)
out,fx,its,imode,smode = slsqp(nfi.f,x0,fprime=nfi.fprime,
        acc=1e-16,iter=15000,iprint=1,full_output=True)
print('\noptimial solution\n',out)



"""
Further more, if we want to impose constraint "nfix[0] = 1e-5" to the problem, 
we can define the projection function of "nfix" and its gradient: "x_proj","grad_proj", 
and then add these hooks by call "nfi.set_options".
"nfi.f" and "nfi.fprime" comes to 
```
class NumpyFunctionInterface:
    def _all_x_proj(self):
        ...
    def _all_grad_proj(self):
        ...
    @property
    def flat_param(self):
        self._all_x_proj()
        ...
    @property.setter
    def flat_param(self,x):
        ...
        self._all_x_proj()
    def fprime(self,x):
        ...
        self._all_grad_proj()
        ...
        return grad
```
"""
def x_proj(params):
    params[0].data[0] = 1e-5
def grad_proj(params):
    params[0].grad.data[0] = 0
## one can also simply set since nfix is globally accessible
# def x_proj(*args,**kw):
#     nfix.data[0] = 1e-5
# def grad_proj(*args,**kw):
#     nfix.grad.data[0] = 0
# nfi.set_oprions(0,x_proj=x_proj,grad_proj=grad_proj)
paramidx = 0
nfi.set_options(paramidx,x_proj=x_proj,grad_proj=grad_proj)
"""
Now we can solve this constraint optimization problem in a unconstraint manner
"""
print("\n\n\n\n ***************** constraint powell_bs ***************** ")
x,f,d = lbfgsb(nfi.f,x0,nfi.fprime,m=100,factr=1,pgtol=1e-14,iprint=10)
out,fx,its,imode,smode = slsqp(nfi.f,x0,fprime=nfi.fprime,
        acc=1e-16,iter=15000,iprint=1,full_output=True)
"""
The original output ('x' or 'out') of the optimizer may not satisfy the constraint.
Recall that the nfi.flat_param will automatically do the projection in reader and setter,
```
class NumpyFunctionInterface:
    @property
    def flat_param(self):
        self._all_x_proj()
        views = []
        for p in self.params:
            views.append(p.view(-1))
        return torch.cat(views,0).numpy()
    @property.setter
    def flat_param(self,x): # x is a numpy array
        for p in self.params: 
            p[:] = x[pidx_start:pidx_end] 
        self._all_x_proj()
```
so we can obtain a constraint gauranteed solution by 
    out = nfi.flat_param
"""
out = nfi.flat_param
print('\noptimial solution\n',out)



"""
Example 2
To further understand "NumpyFunctionInterface", let us extend "powell_bs" 
to a PyTorch custom module 
(see https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_module.html?highlight=custom)
At first, define a pytorch module "penalty=Penalty(100,1e-5)"
"""
import torch.nn as nn
from torch.nn import functional as F
class Penalty(nn.Module):
    def __init__(self,n,alpha=1e-5):
        super(Penalty,self).__init__()
        m = n//2
        x1 = torch.arange(1,m+1).to(torch.float64)
        x2 = torch.arange(m+1,n+1).to(torch.float64)
        self.x1 = nn.Parameter(x1)
        self.x2 = nn.Parameter(x2)
        self.n = n
        self.alpha = alpha
    def forward(self):
        x = torch.cat([self.x1.cpu(),self.x2.cpu()],0)
        return self.alpha*((x-1)**2).sum()+((x**2).sum()-0.25)**2
penalty = Penalty(5,1e-5)
"""
Consider a constraint optimization problem
    min_{penalty.x1,penalty.x2, s.t. penalty.x2[0]=1} penalty.forward()
Then, construct "NumpyFunctionInterface" for this problem (each of the following way is OK)
    # method 0 # penalty.x2 is globally accessible
    def x_proj(*args,**kw):
        penalty.x2.data[0] = 1e-5
    def grad_proj(*args,**kw):
        penalty.x2.grad.data[0] = 0
    nfi = NumpyFunctionInterface(penalty.parameters(),forward=penalty.forward,
        x_proj=x_proj,grad_proj=grad_proj)
    # method 1
    def x_proj(params_of_param_group):
        params_of_param_group[0].data[0] = 1e-5
    def grad_proj(params_of_param_group):
        params_of_param_group[0].grad.data[0] = 0
    nfi = NumpyFunctionInterface([
        dict(params=[penalty.x1,]),
        dict(params=[penalty.x2,],x_proj=x_proj,grad_proj=grad_proj)
        ], penalty.forward)
    # method 2
    def x_proj(params_of_param_group):
        params_of_param_group[1].data[0] = 1e-5
    def grad_proj(params_of_param_group):
        params_of_param_group[1].grad.data[0] = 0
    nfi = NumpyFunctionInterface([
        dict(params=[penalty.x1,penalty.x2],x_proj=x_proj,grad_proj=grad_proj),
        ], penalty.forward)
    # method 3
    def x_proj(params_of_param_group):
        params_of_param_group[1].data[0] = 1e-5
    def grad_proj(params_of_param_group):
        params_of_param_group[1].grad.data[0] = 0
    nfi = NumpyFunctionInterface([penalty.x1,penalty.x2], penalty.forward)
    nfi.set_options(0, x_proj=x_proj, grad_proj=grad_proj)
In "NumpyFunctionInterface", parameters are devided into different parameter groups,
any parameter groups is a dict of 
"""
def x_proj(*args,**kw):
    penalty.x2.data[0] = 1e-5
def grad_proj(*args,**kw):
    penalty.x2.grad.data[0] = 0
nfi = NumpyFunctionInterface(penalty.parameters(),forward=penalty.forward,
    x_proj=x_proj,grad_proj=grad_proj)
# x0 = torch.cat([penalty.x1.cpu(),penalty.x2.cpu()],0).data.clone().numpy()
x0 = np.random.randn(nfi.numel())
print("\n\n\n\n ***************** penalty *****************")
x,f,d = lbfgsb(nfi.f,x0,nfi.fprime,m=100,factr=1,pgtol=1e-14,iprint=10)
out,fx,its,imode,smode = slsqp(nfi.f,x0,fprime=nfi.fprime,acc=1e-16,iter=15000,iprint=1,full_output=True)
# the following two assignments will inforce 'out' to satisfy the constraint
nfi.flat_param = out 
out = nfi.flat_param 
print('\noptimial solution\n',out)
