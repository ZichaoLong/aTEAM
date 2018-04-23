"""interpolatons"""
import numpy as np
from numpy import *
import torch
from torch.autograd import Variable
import torch.nn as nn
from ..functional.interpolation import lagrangeinterp,_ele2coe,_fix_inputs,_base
from ...utils import meshgen

__all__ = ['LagrangeInterp', 'LagrangeInterpFixInputs']

class LagrangeInterp(nn.Module):
    """
    piecewise Lagrange Interpolation in R^m

    Arguments:
        interp_dim (int): spatial dimension, m=interp_dim
        interp_coe (Variable): DoubleTensor (cuda) or FloatTensor (cuda).
            torch.size(mesh_size*interp_degree+1)
        interp_degree (int): degree of Lagrange Interpolation Polynomial
        mesh_bound (ndarray): dtype=double or float. shape=[2,m]. mesh_bound 
            defines the interpolation domain.
        mesh_size (ndarray): dtype=int, shape=[m,]. mesh_size defines the 
            grid number of piecewise interpolation.
    """
    def __init__(self, interp_dim, interp_degree, mesh_bound, mesh_size):
        super(LagrangeInterp, self).__init__()
        self.__m = interp_dim
        self.__d = interp_degree
        self.__mesh_bound = mesh_bound.copy()
        self.__mesh_size = mesh_size.copy()
        __ele2coe = _ele2coe(self.m, self.d)
        __ele2coe = torch.from_numpy(__ele2coe).long()
        self.__ele2coe = __ele2coe 
        # ele2coe should not be registered as buffer
        mesh_size = list(map(lambda x:int(x), list(mesh_size*self.d+1)))
        interp_coe = torch.Tensor(*mesh_size).normal_()
        self.interp_coe = nn.Parameter(interp_coe)

    def init(self, func):
        inputs = meshgen(self.mesh_bound, self.mesh_size*self.d, endpoint=True)
        inputs = torch.from_numpy(inputs)
        inputs = self.interp_coe.data.new(inputs.size()).copy_(inputs)
        self.interp_coe.data = func(inputs)

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
    @property
    def ele2coe(self):
        if self.interp_coe.data.is_cuda:
            device = self.interp_coe.data.get_device()
            self.__ele2coe = self.__ele2coe.cuda(device)
        else:
            self.__ele2coe = self.__ele2coe.cpu()
        return self.__ele2coe

    def forward(self, inputs):
        """
        piecewise Lagrange Interpolation in R^m

        Arguments:
            inputs (Variable): DoubleTensor (cuda) or FloatTensor (cuda). 
                torch.size=[...,m], where m is the spatial dimension.
        """
        size = inputs.size()
        if not inputs.is_contiguous():
            inputs = inputs.clone()
        inputs = inputs.view([-1,self.m])
        outputs = lagrangeinterp(inputs, self.interp_coe, self.m, self.d, 
                self.mesh_bound, self.mesh_size, ele2coe=Variable(self.ele2coe))
        return outputs.view(size[:-1])

class LagrangeInterpFixInputs(LagrangeInterp):
    """
    piecewise Lagrange Interpolation in R^m for fixed inputs.

    Arguments:
        inputs (Variable): DoubleTensor (cuda) or FloatTensor (cuda). 
            torch.size=[...,m], where m is the spatial dimension.
        interp_dim (int): spatial dimension, m=interp_dim
        interp_coe (Variable): DoubleTensor (cuda) or FloatTensor (cuda).
            torch.size(mesh_size*interp_degree+1)
        interp_degree (int): degree of Lagrange Interpolation Polynomial
        mesh_bound (ndarray): dtype=double or float. shape=[2,m]. mesh_bound 
            defines the interpolation domain.
        mesh_size (ndarray): dtype=int, shape=[m,]. mesh_size defines the 
            grid number of piecewise interpolation.
    """
    def __init__(self, inputs, interp_dim, interp_degree, mesh_bound, mesh_size):
        super(LagrangeInterpFixInputs, self).__init__(interp_dim, 
                interp_degree, mesh_bound, mesh_size)
        if not inputs.data.is_cuda:
            self.cpu()
        else:
            self.cuda(inputs.data.get_device())
        if not inputs.data.is_contiguous():
            inputs.data = inputs.data.clone()
        size = inputs.size()
        self.__inputs_size = size
        inputs = inputs.view([-1,self.m])
        self.register_buffer('inputs',inputs.data)
        flat_indices, points_shift = _fix_inputs(inputs, self.m, self.d, \
                mesh_bound, mesh_size, Variable(self.ele2coe))
        self.__flat_indices = flat_indices.data
        self.register_buffer('points_shift', points_shift.data)
        base = _base(points_shift, self.m, self.d)
        self.register_buffer('base', base.data)

    @property
    def flat_indices(self):
        if self.interp_coe.data.is_cuda:
            device = self.interp_coe.data.get_device()
            self.__flat_indices = self.__flat_indices.cuda(device)
        else:
            self.__flat_indices = self.__flat_indices.cpu()
        return self.__flat_indices
    @property
    def inputs_size(self):
        return self.__inputs_size

    def forward(self):
        return lagrangeinterp(Variable(self.inputs), self.interp_coe, 
                self.m, self.d, self.mesh_bound, self.mesh_size, 
                ele2coe=Variable(self.ele2coe), fix_inputs=True, 
                flat_indices=Variable(self.flat_indices), 
                points_shift=Variable(self.points_shift), 
                base=Variable(self.base)).view(self.inputs_size[:-1])

