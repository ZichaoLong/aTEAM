#%%
import torch
import aTEAM
import aTEAM.pdetools.spectral as spectral
import aTEAM.pdetools.init as init
import aTEAM.nn.functional as aF

size = 100
dx = 1/size
u = init.initgen(mesh_size=[size,size], freq=4)
mesh_bound = [[0,0],[1,1]]
# u = u.to(dtype=torch.float32)

upad = aF.periodicpad(u, [0,0,1,1])

u_spect = spectral.time2spect(u, signal_ndim=2)
u10_spect = spectral.spect_diff(u_spect, signal_ndim=2, order=[1,0], mesh_bound=mesh_bound)
u10 = spectral.spect2time(u10_spect, signal_ndim=2)
print(((u10-(upad[2:]-upad[:-2])/(2*dx)).norm()/u10.norm()).item())

#%%
import torch
import aTEAM
import aTEAM.pdetools.spectral as spectral
import aTEAM.pdetools.init as init
import aTEAM.nn.functional as aF

size = 10000
dx = 1/size
u = init.initgen(mesh_size=[size,], freq=3)
mesh_bound = [[0,],[1,]]
# u = u.to(dtype=torch.float32)

upad = aF.periodicpad(u, [1,1])

u_spect = spectral.time2spect(u, signal_ndim=1)
u10_spect = spectral.spect_diff(u_spect, signal_ndim=1, order=[1,], mesh_bound=mesh_bound)
u10 = spectral.spect2time(u10_spect, signal_ndim=1)
print(((u10-(upad[2:]-upad[:-2])/(2*dx)).norm()/u10.norm()).item())
#%%
