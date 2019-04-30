"""interpolatons"""
import numpy as np
from numpy import *
import torch
import torch.nn as nn
from ..functional.interpolation import lagrangeinterp,_ele2coe,_fix_inputs,_base
from ...utils import meshgen

__all__ = ['LagrangeInterp', 'LagrangeInterpFixInputs']

class LagrangeInterp(nn.Module):
    """
    piecewise Lagrange Interpolation in R^m

    Arguments:
        interp_dim (int): spatial dimension, m=interp_dim
        interp_coe (Tensor): DoubleTensor (cuda) or FloatTensor (cuda).
            torch.size(np.array(mesh_size)*interp_degree+1)
        interp_degree (int): degree of Lagrange Interpolation Polynomial
        mesh_bound (tuple): ((l_1,l_2,...,l_n),(u_1,u_2,...,u_n)). mesh_bound 
            defines the interpolation domain. l_i,u_i is lower and upper bound
            of dimension i.
        mesh_size (tuple): mesh_size defines the grid number of 
            piecewise interpolation. mesh_size[i] is mesh num of dimension i.
    """
    def __init__(self, interp_dim, interp_degree, mesh_bound, mesh_size):
        super(LagrangeInterp, self).__init__()
        self.__m = interp_dim
        self.__d = interp_degree
        mesh_bound = array(mesh_bound).reshape(2,self.__m)
        mesh_size = array(mesh_size).reshape(self.__m)
        self.__mesh_bound = mesh_bound.copy()
        self.__mesh_size = mesh_size.copy()
        __ele2coe = _ele2coe(self.m, self.d)
        __ele2coe = torch.from_numpy(__ele2coe).long()
        # nn.Module.to(dtype) will only cast the floating point parameters
        # and buffers to dtype
        self.register_buffer('ele2coe', __ele2coe) 
        mesh_size = list(map(lambda x:int(x), list(mesh_size*self.d+1)))
        interp_coe = torch.Tensor(*mesh_size).normal_()
        self.interp_coe = nn.Parameter(interp_coe)

    def init(self, func, is_numpy_func=False):
        inputs = meshgen(self.mesh_bound, self.mesh_size*self.d, endpoint=True)
        if not is_numpy_func:
            inputs = torch.from_numpy(inputs).to(self.interp_coe)
            self.interp_coe.data = func(inputs)
        else:
            self.interp_coe.data.copy_(torch.from_numpy(func(inputs)))
        return None

    @property
    def m(self):
        return self.__m
    @property
    def d(self):
        return self.__d
    @property
    def mesh_bound(self):
        return self.__mesh_bound
    @property
    def mesh_size(self):
        return self.__mesh_size

    def forward(self, inputs):
        """
        piecewise Lagrange Interpolation in R^m

        Arguments:
            inputs (Tensor): DoubleTensor (cuda) or FloatTensor (cuda). 
                torch.size=[...,m], where m is the spatial dimension.
        """
        size = inputs.size()
        if self.m == 1 and size[-1] != 1:
            inputs = inputs[...,newaxis]
            size = inputs.size()
        inputs = inputs.contiguous()
        inputs = inputs.view([-1,self.m])
        outputs = lagrangeinterp(inputs, self.interp_coe, self.m, self.d, 
                self.mesh_bound, self.mesh_size, ele2coe=self.ele2coe)
        return outputs.view(size[:-1])

class LagrangeInterpFixInputs(LagrangeInterp):
    """
    piecewise Lagrange Interpolation in R^m for fixed inputs.

    Arguments:
        inputs (Tensor): DoubleTensor (cuda) or FloatTensor (cuda). 
            torch.size=[...,m], where m is the spatial dimension.
        interp_dim (int): spatial dimension, m=interp_dim
        interp_coe (Tensor): DoubleTensor (cuda) or FloatTensor (cuda).
            torch.size(np.array(mesh_size)*interp_degree+1)
        interp_degree (int): degree of Lagrange Interpolation Polynomial
        mesh_bound (tuple): ((l_1,l_2,...,l_n),(u_1,u_2,...,u_n)). mesh_bound 
            defines the interpolation domain. l_i,u_i is lower and upper bound
            of dimension i.
        mesh_size (tuple): mesh_size defines the grid number of 
            piecewise interpolation. mesh_size[i] is mesh num of dimension i.
    """
    def __init__(self, inputs, interp_dim, interp_degree, mesh_bound, mesh_size):
        super(LagrangeInterpFixInputs, self).__init__(interp_dim, 
                interp_degree, mesh_bound, mesh_size)
        assert isinstance(inputs, torch.Tensor)
        self.to(dtype=inputs.dtype, device=inputs.device)
        self.register_buffer('_inputs',inputs.new(1))
        self.register_buffer('flat_indices', inputs.new(1).long())
        self.register_buffer('points_shift', inputs.new(1))
        self.register_buffer('base', inputs.new(1))
        self.update_inputs(inputs)

    def update_inputs(self, inputs):
        inputs = inputs.clone()
        inputs.data = inputs.data.to(self._inputs.device)
        inputs.data = inputs.data.contiguous()
        size = inputs.size()
        if self.m == 1 and size[-1] != 1:
            inputs = inputs[...,newaxis]
            size = inputs.size()
        self.__inputs_size = size
        inputs = inputs.view([-1,self.m])
        self._inputs = inputs.data.view(size)
        flat_indices, points_shift = _fix_inputs(inputs, self.m, self.d, \
                self.mesh_bound, self.mesh_size, self.ele2coe)
        self.flat_indices = flat_indices.data
        self.points_shift = points_shift.data
        base = _base(points_shift, self.m, self.d)
        self.base = base.data

    @property
    def inputs(self):
        return self._inputs
    @inputs.setter
    def inputs(self, v):
        self.update_inputs(v)
    @property
    def inputs_size(self):
        return self.__inputs_size

    def forward(self):
        return lagrangeinterp(self._inputs.view([-1,self.m]), self.interp_coe, 
                self.m, self.d, self.mesh_bound, self.mesh_size, 
                ele2coe=self.ele2coe, fix_inputs=True, 
                flat_indices=self.flat_indices, 
                points_shift=self.points_shift, 
                base=self.base).view(self.__inputs_size[:-1])

