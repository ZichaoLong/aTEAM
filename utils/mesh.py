import numpy as np
from numpy import *

__all__ = ['meshgen',]

def meshgen(mesh_bound, mesh_size, endpoint=False):
    mesh_bound = array(mesh_bound, dtype=float64).copy()
    mesh_size = array(mesh_size, dtype=int32).copy()
    if endpoint:
        mesh_bound[1] += (mesh_bound[1]-mesh_bound[0])/mesh_size
        mesh_size += 1
        return meshgen(mesh_bound, mesh_size, endpoint=False)
    N = len(mesh_size)
    xyz = zeros([N,]+list(mesh_size))
    for i in range(N):
        seq = mesh_bound[0,i]+(mesh_bound[1,i]-mesh_bound[0,i])\
                *arange(mesh_size[i])/mesh_size[i]
        newsize = ones(N, dtype=int32)
        newsize[i] = mesh_size[i]
        seq = reshape(seq, newsize)
        xyz[i] = xyz[i]+seq
    perm = arange(1, N+2, dtype=int32)
    perm[N] = 0
    return transpose(xyz, axes=perm)

