#%%
import torch
import aTEAM.nn.modules as am
torch.set_printoptions(precision=2)
shape = [3,15]
order = [0,1]
fd = am.FD.FDProj(kernel_size=shape, order=order, acc=1)
k2m = am.MK.K2M(shape=shape)
m2k = am.MK.M2K(shape=shape)
k2m(fd(fd.subspace[-1].view(*shape)))
a = torch.randn(*shape, dtype=torch.float64)
k2m(fd(a))
#%%
