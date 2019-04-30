"""initial value generator"""
import numpy as np
from numpy import *
import torch

__all__ = ['initgen']

def _initgen_periodic(mesh_size, freq=3):
    dim = len(mesh_size)
    x = random.randn(*mesh_size)
    coe = fft.ifftn(x)
    # set frequency of generated initial value
    freqs = [freq,]*dim
    for i in range(dim):
        perm = arange(dim, dtype=int32)
        perm[i] = 0
        perm[0] = i
        coe = coe.transpose(*perm)
        coe[freqs[i]+1:-freqs[i]] = 0
        coe = coe.transpose(*perm)
    x = fft.fftn(coe)
    assert linalg.norm(x.imag) < 1e-8
    x = x.real
    x = x/np.abs(x).max()
    return x
def _initgen(mesh_size, freq=3, boundary='Periodic', dtype=None, device=None):
    if iterable(freq):
        return freq
    x = _initgen_periodic(mesh_size, freq=freq)
    if boundary.upper() == 'DIRICHLET':
        dim = x.ndim
        for i in range(dim):
            y = arange(mesh_size[i])/mesh_size[i]
            y = y*(1-y)
            s = ones(dim, dtype=int32)
            s[i] = mesh_size[i]
            y = reshape(y, s)
            x = x*y
        x = x[[slice(1,None),]*dim]
        x = x*16
    return torch.from_numpy(x).to(dtype=dtype, device=device)
def initgen(mesh_size, freq=3, boundary='Periodic', dtype=None, device=None, batch_size=1):
    xs = []
    for k in range(batch_size):
        xs.append(_initgen(mesh_size, freq=freq, boundary=boundary, dtype=dtype, device=device))
    x = torch.stack(xs, dim=0)
    if batch_size == 1:
        return x[0]
    else:
        return x
