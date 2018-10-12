import numpy as np
import torch
import torch.nn as nn

__all__ = ['spect_filter','time2spect','spect2time','SpectMul','spect_diff']

def _spect_filter(spect0, spect_size=None, axis=-2):
    if spect_size is None:
        return spect0
    assert isinstance(spect_size, int) and spect_size%2 == 0 and spect_size > 2
    assert axis != -1 and axis != spect0.dim()-1
    shape = list(spect0.shape)
    size = shape[axis]
    gap = spect_size-size
    if gap == 0:
        return spect0
    assert shape[-1] == 2 and size%2 == 0 and size > 2
    idx1 = [slice(None),]*spect0.dim()
    idx2 = [slice(None),]*spect0.dim()
    N = (spect_size//2 if gap<0 else size//2)
    idx1[axis] = slice(0,N)
    idx2[axis] = slice(-N+1,None)
    shape[axis] = (1 if gap<0 else gap+1)
    z = torch.zeros(*shape, dtype=spect0.dtype, device=spect0.device)
    return torch.cat([spect0[idx1],z,spect0[idx2]], dim=axis)

def spect_filter(spect0, spect_size=None):
    if spect_size is None:
        return spect0
    if isinstance(spect_size, int):
        spect_size = [spect_size,]
    axis = list(range(-len(spect_size)-1,-1))
    for s,a in zip(spect_size, axis):
        spect0 = _spect_filter(spect0, s, axis=a)
    return spect0

def time2spect(u, signal_ndim, spect_size=None):
    if u.shape[-1] != 2:
        z = torch.zeros_like(u)
        u = torch.stack([u,z], dim=-1)
    spect = torch.ifft(u, signal_ndim)
    if not spect_size is None:
        assert len(spect_size) == signal_ndim
    return spect_filter(spect, spect_size)
def spect2time(spect, signal_ndim, time_size=None, isreal=True):
    if not time_size is None:
        assert len(time_size) == signal_ndim
    spect = spect_filter(spect, time_size)
    u = torch.fft(spect, signal_ndim)
    if isreal:
        u = u[...,0]
    return u

def SpectMul(u, v, signal_ndim):
    if u.shape[-1] == 2:
        isreal = False
    else:
        isreal = True
    u = time2spect(u, signal_ndim)
    v = time2spect(v, signal_ndim)
    size = list(u.shape[i] for i in range(-signal_ndim-1,-1))
    spect_size = list(3*u.shape[i]//2 for i in range(-signal_ndim-1,-1))
    u = spect_filter(u, spect_size)
    v = spect_filter(v, spect_size)
    u = spect2time(u, signal_ndim, isreal=False)
    v = spect2time(v, signal_ndim, isreal=False)
    c = torch.stack(
            [v[...,0]*u[...,0]-v[...,1]*u[...,1],
                v[...,1]*u[...,0]+v[...,0]*u[...,1]],
            dim=-1
            )
    c = time2spect(c, signal_ndim, size)
    c = spect2time(c, signal_ndim, isreal=isreal)
    return c
class SpectMulFixV(nn.Module):
    def __init__(self, v, signal_ndim):
        if v.shape[-1] != 2:
            z = torch.zeros_like(v)
            v = torch.stack([v,z], dim=-1)
        self.size = list(v.shape[i] for i in range(-signal_ndim-1,-1))
        self.signal_ndim = signal_ndim
        self.spect_size = list(3*v.shape[i]//2 for i in range(-signal_ndim-1,-1))
        _v = time2spect(v, self.signal_ndim, self.spect_size)
        _v = spect2time(_v, self.signal_ndim)
        self.register_buffer('aug_v', _v)
    def forward(self, u, isreal=True, spect_or_real='real'):
        if spect_or_real is None:
            spect_or_real = 'spect'
        if spect_or_real.upper() == 'REAL':
            u = time2spect(u, self.signal_ndim, self.spect_size)
        else:
            assert u.shape[-1] == 2
            u = spect_filter(u, self.spect_size)
        u = spect2time(u, self.signal_ndim)
        v = self.aug_v
        c = torch.stack(
                [v[...,0]*u[...,0]-v[...,1]*u[...,1],
                    v[...,1]*u[...,0]+v[...,0]*u[...,1]],
                dim=-1
                )
        c = time2spect(c, self.signal_ndim, self.size)
        c = spect2time(c, self.signal_ndim, isreal=isreal)
        return c

def spect_diff(u_spect, signal_ndim, order, mesh_bound=None):
    size0 = u_spect.shape
    s = [1,]*u_spect.dim()
    freq0 = np.ones(s)
    assert len(order) == signal_ndim
    b = u_spect.dim()-signal_ndim-1
    for i in range(signal_ndim):
        if order[i] == 0:
            continue
        freq = np.fft.fftfreq(size0[b+i], 1/size0[b+i])
        if not mesh_bound is None:
            freq *= 2*np.pi/(mesh_bound[1][i]-mesh_bound[0][i])
        freq = freq**order[i]
        s[b+i] = -1
        freq = np.reshape(freq, s)
        s[b+i] = 1
        freq0 = freq0*freq
    freq0 = torch.from_numpy(freq0).to(u_spect)
    u_spect = u_spect*freq0
    totalorder = sum(order)
    if totalorder%4 == 1:
        u_spect = torch.stack([u_spect[...,1],-u_spect[...,0]], dim=-1)
    elif totalorder%4 == 2:
        u_spect = -u_spect
    elif totalorder%4 == 3:
        u_spect = torch.stack([-u_spect[...,1],u_spect[...,0]], dim=-1)
    return u_spect
