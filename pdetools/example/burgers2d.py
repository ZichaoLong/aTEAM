import numpy as np
import torch
import torch.nn as nn
from ..stepper import TimeStepper
from ..upwind import UpWind2dRHI
from ..init import initgen
from ..spectral import *

class BurgersTime2d(nn.Module, TimeStepper):
    """
    2d Burgers equation
    \partial_t u+u\cdot\nabla u=v\laplace u+f
    """
    @property
    def timescheme(self):
        return self._timescheme
    @property
    def spatialscheme(self):
        return self._spatialscheme
    def RightHandItems(self, u, viscosity=None, force=None, **kw):
        """
        u[...,0,y,x],u[...,1,y,x]
        """
        coe = np.ndarray((3,3), dtype=object)
        coe[0,0] = coe[1,1] = 0
        coe[0,2] = coe[2,0] = (self.viscosity if viscosity is None else viscosity)
        coe[0,1] = u[...,:1,:,:]
        coe[1,0] = u[...,1:,:,:]
        rhi = UpWind2dRHI(self.dx, coe, u, self.spatialscheme)
        if not force is None:
            rhi = rhi+force
        elif not self.force is None:
            rhi = rhi+self.force
        return rhi
    def __init__(self, max_dt, mesh_size, mesh_bound=((0,0),(1,1)), viscosity=0.01, timescheme='rk2', spatialscheme='uw2', force=None):
        super(BurgersTime2d, self).__init__()
        self.max_dt = max_dt
        self.mesh_size = np.array(mesh_size).copy()
        self.mesh_bound = np.array(mesh_bound).copy()
        dx0,dx1 = (self.mesh_bound[1]-self.mesh_bound[0])/self.mesh_size
        assert abs(dx0-dx1)<1e-10
        self.dx = dx0
        self.viscosity = viscosity
        self._timescheme = timescheme
        self._spatialscheme = spatialscheme
        self.force = force
    def forward(self, inputs, T, **kw):
        return self.predict(inputs, T, **kw)
class BurgersSpect2d(nn.Module, TimeStepper):
    """
    2d Burgers equation
    \partial_t u+u\cdot\nabla u=v\laplace u+f
    """
    @property
    def timescheme(self):
        return self._timescheme
    def RightHandItems(self, u, viscosity=None, force=None, **kw):
        """
        u[...,0,y,x],u[...,1,y,x]
        """
        coe = np.ndarray((3,3), dtype=object)
        coe[0,1] = u[...,:1,:,:]
        coe[1,0] = u[...,1:,:,:]
        vis = (viscosity if not viscosity is None else self.viscosity)
        u_spect = time2spect(u, signal_ndim=2)
        tmp = spect_diff(u_spect, signal_ndim=2, order=(0,1), mesh_bound=self.mesh_bound)
        rhi = -SpectMul(coe[0,1], spect2time(tmp, signal_ndim=2, isreal=True), signal_ndim=2)
        tmp = spect_diff(u_spect, signal_ndim=2, order=(1,0), mesh_bound=self.mesh_bound)
        rhi = rhi-SpectMul(coe[1,0], spect2time(tmp, signal_ndim=2, isreal=True), signal_ndim=2)

        rhi = rhi+vis*spect2time(spect_diff(u_spect, signal_ndim=2, order=(0,2), mesh_bound=self.mesh_bound)
                +spect_diff(u_spect, signal_ndim=2, order=(2,0), mesh_bound=self.mesh_bound),
                signal_ndim=2, isreal=True)
        if not force is None:
            rhi = rhi+force
        elif not self.force is None:
            rhi = rhi+self.force
        return rhi
    def __init__(self, max_dt, mesh_size, mesh_bound=((0,0),(1,1)), viscosity=0.01, timescheme='rk2', force=None):
        super(BurgersSpect2d, self).__init__()
        self.max_dt = max_dt
        self.mesh_size = np.array(mesh_size).copy()
        self.mesh_bound = np.array(mesh_bound).copy()
        dx0,dx1 = (self.mesh_bound[1]-self.mesh_bound[0])/self.mesh_size
        assert abs(dx0-dx1)<1e-10
        self.dx = dx0
        self.viscosity = viscosity
        self._timescheme = timescheme
        self.force = force
    def forward(self, inputs, T, **kw):
        return self.predict(inputs, T, **kw)


def test_Burgers2d(viscosity=0.1, max_dt=1e-5):
    import aTEAM.pdetools as pdetools
    import aTEAM.pdetools.example.burgers2d as burgers2d
    import torch
    import matplotlib.pyplot as plt
    import time
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = None
    mesh_size = [64,64]
    T = 1e-3
    batch_size = 2

    init = pdetools.init.initgen(mesh_size=mesh_size, freq=1, device=device, batch_size=2*batch_size)*0.5
    init += init.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]*\
            torch.randn(2*batch_size,1,1, dtype=torch.float64, device=device)*\
            torch.rand(2*batch_size,1,1, dtype=torch.float64, device=device)*2
    burgers0 = burgers2d.BurgersTime2d(max_dt=max_dt, mesh_size=mesh_size, mesh_bound=((0,0),(1,1)), viscosity=viscosity, timescheme='rk2', spatialscheme='uw2')
    burgers1 = burgers2d.BurgersSpect2d(max_dt=max_dt, mesh_size=mesh_size, mesh_bound=((0,0),(1,1)), viscosity=viscosity, timescheme='rk2')
    h = plt.figure()
    stream0 = h.add_subplot(2,3,1,aspect='equal')
    xdiffax = h.add_subplot(2,3,4,aspect='equal')
    a0 = h.add_subplot(2,3,2,aspect='equal')
    b0 = h.add_subplot(2,3,3,aspect='equal')
    a1 = h.add_subplot(2,3,5,aspect='equal')
    b1 = h.add_subplot(2,3,6,aspect='equal')
    def resetticks(*argv):
        for par in argv:
            par.set_xticks([]); par.set_yticks([])
    resetticks(a0,b0,a1,b1)
    x0 = init.view([batch_size,2,]+mesh_size)
    x1 = x0

    Y,X = np.mgrid[0:1:(mesh_size[0]+1)*1j,0:1:(mesh_size[1]+1)*1j]
    Y,X = Y[:-1,:-1],X[:-1,:-1]
    for i in range(20):
        stream0.clear(); xdiffax.clear()
        a0.clear(); a1.clear()
        b0.clear(); b1.clear()

        speed0 = torch.sqrt(x0[0,0]**2+x0[0,1]**2).data.cpu().numpy()
        stream0.streamplot(X,Y,x0[0,0],x0[0,1],density=0.8,color='k',linewidth=5*speed0/speed0.max())
        timea0 = a0.imshow(x0[0,0].data.cpu().numpy()[::-1], cmap='jet')
        timeb0 = b0.imshow(x0[0,1].data.cpu().numpy()[::-1], cmap='jet')
        timec0 = h.colorbar(timea0, ax=a0)
        timed0 = h.colorbar(timeb0, ax=b0)
        resetticks(a0,b0)
        stream0.set_title('max-min(speed)={:.2f}'.format(speed0.max()-speed0.min()))

        xdiff = torch.sqrt((x0[0,0]-x1[0,0])**2+(x0[0,1]-x1[0,1])**2)
        xdiffim = xdiffax.imshow(xdiff.data.cpu().numpy()[::-1],cmap='jet')
        specta1 = a1.imshow(x1[0,0].data.cpu().numpy()[::-1], cmap='jet')
        spectb1 = b1.imshow(x1[0,1].data.cpu().numpy()[::-1], cmap='jet')
        spectc1 = h.colorbar(specta1, ax=a1)
        spectd1 = h.colorbar(spectb1, ax=b1)
        resetticks(a1,b1,xdiffax)
        xdiffax.set_title('max={:.2f},min={:.2f}'.format(xdiff.max().item(),xdiff.min().item()))

        h.suptitle('t={:.1e}'.format(i*T))

        speedrange = max(x0[0,0].max().item()-x0[0,0].min().item(),x0[0,1].max().item()-x0[0,1].min().item())
        relsolutiondiff = (x0-x1).abs().max().item()/speedrange

        startt = time.time()
        with torch.no_grad():
            x0 = burgers0.predict(x0, T=T)
            x1 = burgers1.predict(x1, T=T)
        stopt = time.time()
        print('elapsed-time:{:.1f}'.format(stopt-startt)+
                ', speedrange:{:.0f}'.format(speedrange)+
                ', relsolutiondiff:{:.4f}'.format(relsolutiondiff)
                )
        if i > 0:
            timec0.remove()
            timed0.remove()
            spectc1.remove()
            spectd1.remove()
        plt.pause(1e-3)
