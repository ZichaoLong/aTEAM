import numpy as np
import torch
from ..nn import functional as aF

__all__ = ['UpWind1dRHI', 'UpWind2dRHI', 'UpWind3dRHI']

def _pad(inputs, pad, mode='wrap', value=0):
    if mode == 'wrap':
        return aF.periodicpad(inputs, pad)
    else:
        return torch.nn.functional.pad(inputs, pad, mode, value)

def _UpWind1dRHI(dx, coe, u, spatialscheme, axis, mode='wrap'):
    """
    broadcasting between coe[i] and u
    """
    ndim = u.dim()
    for i in range(len(coe)):
        if not isinstance(coe[i], torch.Tensor):
            coe[i] = torch.from_numpy(np.array(coe[i])).to(u)
    pad_width = [0,0,]*ndim
    idx = [slice(None),]*ndim
    z = torch.zeros(1).to(u)
    if spatialscheme == 'uw1':
        # 一阶迎风
        pad_width[2*axis] = 1
        pad_width[2*axis+1] = 1
        pad_width.reverse()
        u_pad = _pad(u, pad_width, mode=mode)
        idx1 = idx.copy()
        idx2 = idx.copy()
        idx1[axis] = slice(2,None)
        idx2[axis] = slice(None,-2)
        rhi = torch.max(z,-coe[1])*(u_pad[idx1]-u)/dx
        rhi = rhi+torch.min(z,-coe[1])*(u-u_pad[idx2])/dx
        rhi = rhi+coe[2]/dx**2*(u_pad[idx1]+u_pad[idx2]-2*u)
    elif spatialscheme == 'uw2':
        # 二阶迎风
        pad_width[2*axis] = 2
        pad_width[2*axis+1] = 2
        pad_width.reverse()
        u_pad = _pad(u, pad_width, mode=mode)
        idx1 = idx.copy()
        idx2 = idx.copy()
        idx3 = idx.copy()
        idx4 = idx.copy()
        idx1[axis] = slice(3,-1)
        idx2[axis] = slice(4,None)
        idx3[axis] = slice(1,-3)
        idx4[axis] = slice(None,-4)
        rhi = torch.max(z,-coe[1])*(u_pad[idx1]-u_pad[idx2]+3*(u_pad[idx1]-u))/(2*dx)
        rhi = rhi+torch.min(z,-coe[1])*(u_pad[idx4]-u_pad[idx3]+3*(u-u_pad[idx3]))/(2*dx)
        rhi = rhi+coe[2]/dx**2*(u_pad[idx1]+u_pad[idx3]-2*u)
    return rhi

def UpWind1dRHI(dx, coe, u, spatialscheme, mode='wrap'):
    return _UpWind1dRHI(dx, coe, u, spatialscheme, axis=-1, mode=mode)

def UpWind2dRHI(dx, coe, u, spatialscheme, mode='wrap'):
    rhi1 =  _UpWind1dRHI(dx, [0,coe[0,1],coe[0,2]], u, spatialscheme, axis=-1, mode=mode)
    rhi2 =  _UpWind1dRHI(dx, [0,coe[1,0],coe[2,0]], u, spatialscheme, axis=-2, mode=mode)
    # rhi_11 = coe[1,1]
    return rhi1+rhi2

def UpWind3dRHI(dx, coe, u, spatialscheme):
    rhi1 = _UpWind1dRHI(dx, [0,coe[0,0,1],coe[0,0,2]], u, spatialscheme, axis=-1, mode=mode)
    rhi2 = _UpWind1dRHI(dx, [0,coe[0,1,0],coe[0,2,0]], u, spatialscheme, axis=-2, mode=mode)
    rhi3 = _UpWind1dRHI(dx, [0,coe[1,0,0],coe[2,0,0]], u, spatialscheme, axis=-3, mode=mode)
    # rhi_011 = coe[0,1,1]
    # rhi_101 = coe[1,0,1]
    # rhi_110 = coe[1,1,0]
    return rhi1+rhi2+rhi3

