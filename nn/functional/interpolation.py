"""interpolatons"""
import numpy as np
from numpy import *
import torch
import torch.nn.functional as F
from scipy.special import factorial

__all__ = ['lagrangeinterp',]

def _ele2coe(m, degree):
    """
    Arguments:
        m (int): interpolation dimension
        degree (int): degree of Lagrange Interpolation Polynomial
    Returns:
        ele2coe (ndarray): dtype=int64, shape=[degree+1,...,degree+1,m]
            for a_i in 0,...,degree, 
                ele2coe[a_1,a_2,...,a_m] = array([a_1,...,a_m])
    """
    ele2coe = zeros([degree+1,]*m+[m,])
    perm = arange(m+1)
    perm[1:] = arange(m)
    perm[0] = m
    ele2coe = transpose(ele2coe, axes=perm)
    for i in range(m):
        perm = arange(m+1)
        perm[1] = i+1
        perm[i+1] = 1
        ele2coe = transpose(ele2coe, axes=perm)
        for j in range(degree+1):
            ele2coe[i,j] = j
        ele2coe = transpose(ele2coe, axes=perm)
    perm = arange(m+1)
    perm[:m] = arange(1, m+1)
    perm[m] = 0
    ele2coe = transpose(ele2coe, axes=perm)
    return ele2coe

def _fix_inputs(inputs, interp_dim, interp_degree, 
        mesh_bound, mesh_size, ele2coe):
    """
    Arguments:
        inputs (Tensor): DoubleTensor (cuda) or FloatTensor (cuda). 
            torch.size=[N,m], where N is the number of points which will be 
            interpolated, m is the spatial dimension.
        interp_dim (int): spatial dimension, m=interp_dim
        interp_degree (int): degree of Lagrange Interpolation Polynomial
        mesh_bound (ndarray): dtype=double or float. shape=[2,m]. mesh_bound 
            defines the interpolation domain.
        mesh_size (ndarray): dtype=int, shape=[m,]. mesh_size defines the 
            grid number of piecewise interpolation.
        ele2coe (Tensor): see lagrangeinterp.
    Returns:
        flat_indices (Tensor)
        points_shift (Tensor)
    """
    N = inputs.size()[0]
    m = interp_dim
    d = interp_degree

    mesh_bound = torch.from_numpy(mesh_bound).to(inputs)
    mesh_size = torch.tensor(mesh_size, dtype=torch.int64).to(inputs.device)
    inputs = torch.max(inputs, mesh_bound[newaxis,0,:])
    inputs = torch.min(inputs, mesh_bound[newaxis,1,:])
    mesh_delta = (mesh_bound[1]-mesh_bound[0])/mesh_size.type_as(mesh_bound)
    points_shift = (inputs-mesh_bound[newaxis,0,:])/mesh_delta[newaxis]

    element_indices = torch.floor(points_shift.data)

    element_indices = F.relu(element_indices)
    supindices = mesh_size[newaxis,:].type_as(mesh_bound)-1
    element_indices = supindices-F.relu(supindices-element_indices)
    points_shift = points_shift-element_indices
    element_indices = element_indices.long()
    # element_indices.size(): [N,m], 
    # 0 <= element_indices[i] <= mesh_size-1, i=0,...,N-1

    interp_coe_indices = (element_indices*d).\
            view([N,]+[1,]*m+[m,])+ele2coe[newaxis]
    # interp_coe_indices.size(): [N,]+[d+1,...,d+1]+[m,]
    interp_coe_indices = interp_coe_indices.view([-1,m])
    flat_indices = element_indices.data.new(N*(d+1)**m).zero_()
    base = 1
    for i in range(m-1,-1,-1):
        flat_indices += interp_coe_indices[:,i]*base
        base *= mesh_size.data[i]*d+1
    return flat_indices, points_shift

def _base(points_shift, interp_dim, interp_degree):
    """
    Arguments:
        points_shift (Tensor): DoubleTensor (cuda) or FloatTensor (cuda). 
            torch.size=[N,m], where N is the number of points which will be 
            interpolated, m is the spatial dimension.
        interp_dim (int): spatial dimension, m=interp_dim
        interp_degree (int): degree of Lagrange Interpolation Polynomial
    Returns:
        base (Tensor)
    """
    N = points_shift.size()[0]
    m = interp_dim
    d = interp_degree

    base_function = ndarray(shape=[m,d+1],dtype=np.object)
    grid = torch.from_numpy(arange(d+1)/d)[newaxis,:].to(points_shift)
    for i in range(m):
        M = points_shift[:,i,newaxis]-grid
        for j in range(d+1):
            M1 = torch.prod(M[:,:j], dim=1) if j!=0 else 1
            M2 = torch.prod(M[:,j+1:], dim=1) if j!=d else 1
            base_function[i,j] = \
                    M1*M2*(d**d/factorial(j)/factorial(d-j)*(-1)**(d-j))

    # base = points_shift.data.new(1).fill_(1)
    for i in range(m):
        base_tmp0 = [0,]*(d+1)
        for j in range(d+1):
            base_tmp0[j] = base_function[i,j][:,newaxis]
        base_tmp1 = torch.cat(base_tmp0, dim=1).view([N,]+[1,]*i+[d+1,])
        if i == 0:
            base = base_tmp1
        else:
            base = base[...,newaxis]*base_tmp1
    base = base.view([N,-1])
    return base

def lagrangeinterp(inputs, interp_coe, interp_dim, interp_degree, 
        mesh_bound, mesh_size, *, ele2coe=None, 
        fix_inputs=False, flat_indices=None, points_shift=None, 
        base=None):
    """
    piecewise Lagrange Interpolation in R^m

    Arguments:
        inputs (Tensor): DoubleTensor (cuda) or FloatTensor (cuda). 
            torch.size=[N,m], where N is the number of points which will be 
            interpolated, m is the spatial dimension.
        interp_dim (int): spatial dimension, m=interp_dim
        interp_coe (Tensor): DoubleTensor (cuda) or FloatTensor (cuda).
            torch.size(np.array(mesh_size)*interp_degree+1)
        interp_degree (int): degree of Lagrange Interpolation Polynomial
        mesh_bound (tuple): ((l_1,l_2,...,l_n),(u_1,u_2,...,u_n)). mesh_bound 
            defines the interpolation domain. l_i,u_i is lower and upper bound
            of dimension i.
        mesh_size (tuple): mesh_size defines the grid number of 
            piecewise interpolation. mesh_size[i] is mesh num of dimension i.
    Returns:
        outputs (Tensor): torch.size=[N,], interpolation value of inputs 
            using interp_coe.
    """
    inputs = inputs.contiguous()
    inputs = inputs.view([-1,interp_dim])
    N = inputs.size()[0]
    m = interp_dim
    d = interp_degree
    assert d>0, "degree of interpolation polynomial must be greater than 0."
    mesh_bound = array(mesh_bound).reshape(2,m)
    mesh_size = array(mesh_size).reshape(m)

    assert inputs.device == interp_coe.device

    if ele2coe is None:
        ele2coe = torch.from_numpy(_ele2coe(m, d)).long().to(inputs.device)

    if not fix_inputs:
        flat_indices, points_shift = _fix_inputs(inputs, m, d, \
                mesh_bound, mesh_size, ele2coe)
    
    interp_coe = interp_coe.contiguous()
    interp_coe_resp = torch.gather(interp_coe.view([-1,]), 0, flat_indices)
    interp_coe_resp = interp_coe_resp.view([N,-1])

    if not fix_inputs:
        base = _base(points_shift, m, d)

    outputs = (interp_coe_resp*base).sum(dim=1)
    return outputs
