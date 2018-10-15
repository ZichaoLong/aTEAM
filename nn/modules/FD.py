"""Finite Difference tools"""
import numpy as np
from numpy import *
from numpy.linalg import *
from functools import reduce
from scipy.special import factorial
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import MK
from ..functional import periodicpad

__all__ = ['FDMK','FD1d','FD2d','FD3d','FDProj']

def _inv_equal_order_m(d,m):
    A = []
    assert d >= 1 and m >= 0
    if d == 1:
        A = [[m,],]
        return A
    if m == 0:
        for i in range(d):
            A.append(0)
        return [A,]
    for k in range(m+1):
        B = _inv_equal_order_m(d-1,m-k)
        for b in B:
            b.append(k)
        A = A+B
    return A

def _less_order_m(d,m):
    A = []
    for k in range(m+1):
        B = _inv_equal_order_m(d,k)
        for b in B:
            b.reverse()
        B.sort()
        B.reverse()
        A.append(B)
    return A

class FDMK(nn.Module):
    """
    Moment matrix and kernel for finite difference.
    Arguments:
        dim (int): dimension
        kernel_size (tuple of int): size of differential kernels
        order (tuple of int): order of differential kernels
        dx (double): the MomentBank.kernel will automatically compute kernels 
            according to MomentBank.moment and MomentBank.dx
        constraint (string): 'moment' or 'free', See FDMK.x_proj 
            and FDMK.grad_proj.
    """
    def __init__(self, dim, kernel_size, order, dx=1.0, constraint='moment'):
        super(FDMK, self).__init__()
        self._dim = dim
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size,]*self.dim
        assert min(kernel_size) > max(order)
        self.m2k = MK.M2K(kernel_size)
        self.k2m = MK.K2M(kernel_size)
        self._kernel_size = tuple(kernel_size)
        self._order = order
        self.constraint = constraint

        scale = torch.DoubleTensor(1)[0]
        self.register_buffer('scale',scale)
        if not iterable(dx):
            dx = [dx,]*dim
        self.dx = dx.copy()

        self._order_bank = _less_order_m(dim, max(kernel_size))
        moment = torch.DoubleTensor(*kernel_size).zero_()
        moment[tuple(self._order)] = 1
        self.moment = nn.Parameter(moment)

    @property
    def dim(self):
        return self._dim
    @property
    def dx(self):
        return self._dx.copy()
    @dx.setter
    def dx(self,v):
        """
        v (ndarray): dx for each axis
        """
        if not iterable(v):
            v = [v,]*self.dim
        self._dx = v
        l = lambda a,b:a*b
        s = reduce(l, (self.dx[j]**oj for j,oj in enumerate(self._order)), 1)
        self.scale.fill_(1/s)
        return v
    @property
    def kernel(self):
        kernel = self.m2k(self.moment)
        return kernel
    @kernel.setter
    def kernel(self, v):
        if isinstance(v, (list,tuple)):
            v = np.array(v)
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        if isinstance(v, torch.Tensor):
            v = v.to(self.moment)
        moment = self.k2m(v)
        self.moment.data.copy_(moment)
        return self.moment

    def _proj_(self,M,s,c):
        for j in range(s):
            for o in self._order_bank[j]:
                M[tuple(o)] = c
    def x_proj(self,*args,**kw):
        if self.constraint == 'free':
            return None
        if isinstance(self.constraint,int):
            acc = self.constraint
        else:
            acc = 1
        self._proj_(self.moment.data,sum(self._order)+acc,0)
        self.moment.data[tuple(self._order)] = 1
        return None
    def grad_proj(self,*args,**kw):
        if self.constraint == 'free':
            return None
        if isinstance(self.constraint,int):
            acc = self.constraint
        else:
            acc = 1
        self._proj_(self.moment.grad.data,sum(self._order)+acc,0)
        return None

    def forward(self):
        raise NotImplementedError

class _FDNd(FDMK):
    """
    Finite difference automatically handle boundary conditions
    Arguments for class:`_FDNd`:
        dim (int): dimension
        kernel_size (tuple of int): finite difference kernel size
        boundary (string): 'Dirichlet' or 'Periodic'
    Arguments for class:`FDMK`:
        order, dx, constraint
    """
    def __init__(self, dim, kernel_size, order, 
            dx=1.0, constraint='moment', boundary='Dirichlet'):
        super(_FDNd, self).__init__(dim, kernel_size, order, dx, constraint)
        padwidth = []
        for k in reversed(self._kernel_size):
            padwidth.append((k-1)//2)
            padwidth.append(k-1-(k-1)//2)
        self._padwidth = padwidth
        self.boundary = boundary.upper()

    @property
    def padwidth(self):
        return self._padwidth.copy()
    @property
    def boundary(self):
        return self._boundary
    @boundary.setter
    def boundary(self,v):
        self._boundary = v.upper()
    def pad(self, inputs):
        if self.boundary == 'DIRICHLET':
            return F.pad(inputs, self.padwidth)
        else:
            return periodicpad(inputs, self.padwidth)

    def conv(self, inputs, weight):
        raise NotImplementedError
    def forward(self, inputs, kernel=None, scale=None):
        """
        Arguments:
            inputs (Tensor): torch.size: 
                (batch_size, spatial_size[0], spatial_size[1], ...)
            kernel (Tensor): torch.size: 
                (kernel_size[0], kernel_size[1], ...)
            scale (scalar): depends on self.dx
        Returns:
            approximation of self.order partial derivative of inputs
        """
        scale = (self.scale if scale is None else scale)
        kernel = (self.kernel if kernel is None else kernel)
        kernel = kernel*scale
        assert inputs.dim() == kernel.dim()+1
        inputs = self.pad(inputs)
        inputs = inputs[:,newaxis]
        return self.conv(inputs, kernel[newaxis,newaxis])[:,0]

class FD1d(_FDNd):
    def __init__(self, kernel_size, order, 
            dx=1.0, constraint='moment', boundary='Dirichlet'):
        if isinstance(order, int):
            order = (order,)
        super(FD1d, self).__init__(1, kernel_size, order, 
                dx=dx, constraint=constraint, boundary=boundary)
        self.conv = F.conv1d
class FD2d(_FDNd):
    def __init__(self, kernel_size, order, 
            dx=1.0, constraint='moment', boundary='Dirichlet'):
        super(FD2d, self).__init__(2, kernel_size, order, 
                dx=dx, constraint=constraint, boundary=boundary)
        self.conv = F.conv2d
class FD3d(_FDNd):
    def __init__(self, kernel_size, order, 
            dx=1.0, constraint='moment', boundary='Dirichlet'):
        super(FD3d, self).__init__(3, kernel_size, order, 
                dx=dx, constraint=constraint, boundary=boundary)
        self.conv = F.conv3d

class FDProj(nn.Module):
    """
    project convolution kernel to finite difference coefficient
    """
    def __init__(self, kernel_size, order, acc=1):
        super(FDProj, self).__init__()
        assert sum(order)<min(kernel_size)
        self.dim = len(kernel_size)
        self.n = 1
        for i in kernel_size:
            self.n *= i
        self.order = order
        self.m = sum(order)
        m = self.m+acc-1
        self._order_bank = _less_order_m(self.dim, m)
        s = [1,]*self.dim
        base = []
        for i in range(self.dim):
            b = torch.arange(kernel_size[i], dtype=torch.float64)-(kernel_size[i]-1)//2
            s[i] = -1
            b = b.view(*s)
            s[i] = 1
            base.append(b)
        subspaces = []
        for j in range(m+1):
            for o in self._order_bank[j]:
                b = torch.ones(*kernel_size, dtype=torch.float64)
                for i in range(self.dim):
                    if o[i]>0:
                        b *= base[i]**o[i]
                b = b.view(-1)
                if tuple(o) == tuple(order):
                    subspaces.insert(0,b)
                    continue
                subspaces.append(b)
        subspaces.reverse()
        l = len(subspaces)
        # Schimidt orthogonalization
        for i in range(l):
            for j in range(i):
                subspaces[i] -= torch.dot(subspaces[j], subspaces[i])*subspaces[j]
            subspaces[i] = subspaces[i]/torch.sqrt(torch.dot(subspaces[i], subspaces[i]))
        subspace = torch.stack(subspaces, dim=0)
        self.register_buffer('subspace', subspace)
        moment = torch.ones(*kernel_size, dtype=torch.float64)
        for i in range(self.dim):
            if order[i]>0:
                moment *= base[i]**order[i]/factorial(order[i]).item()
        moment = moment.view(-1)
        self.register_buffer('_renorm', 1/torch.dot(moment,subspace[-1]))
    def forward(self, kernel):
        shape = kernel.shape
        kernel = kernel.contiguous()
        kernel = kernel.view(-1,self.n)
        kernel = kernel-kernel@self.subspace.transpose(0,1)@self.subspace
        kernel = kernel+self._renorm*self.subspace[-1:]
        kernel = kernel.view(shape)
        return kernel
