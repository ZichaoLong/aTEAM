import numpy as np
import torch
import torch.nn as nn
from ..stepper import LinearTimeStepper, LinearSpectStepper
from ..init import initgen

__all__ = ['CDE', 'SingleCell1', 'SingleCell2']

# Convection diffusion equation
class _CDE(nn.Module):
    @property
    def dim(self):
        return 2
    @property
    def timescheme(self):
        return self._timescheme
    @property
    def spatialscheme(self):
        return self._spatialscheme
    @property
    def coe(self):
        return self._coe
    def setcoe(self):
        raise NotImplementedError
    def __init__(self, max_dt=0.2e-3, mesh_size=(256,256), mesh_bound=((0,0),(1,1))):
        super(_CDE, self).__init__()
        self.max_dt = max_dt
        self.mesh_size = np.array(mesh_size).copy()
        self.mesh_bound = np.array(mesh_bound).copy()
        dx0,dx1 = (self.mesh_bound[1]-self.mesh_bound[0])/self.mesh_size
        assert abs(dx0-dx1)<1e-10
        self.dx = dx0
        xy = np.mgrid[self.mesh_bound[0][0]:self.mesh_bound[1][0]:(self.mesh_size[0]+1)*1j,
                self.mesh_bound[0][1]:self.mesh_bound[1][1]:(self.mesh_size[1]+1)*1j]
        xy = xy[:,:-1,:-1]
        xy = np.transpose(xy, axes=[1,2,0])
        xy = torch.from_numpy(xy)
        self.setcoe(xy)
    def forward(self, inputs, T, **kw):
        return self.predict(inputs, T, **kw)

class CDE(_CDE, LinearTimeStepper):
    def setcoe(self, xy):
        coe = np.ndarray((3,3), dtype=object)
        self._coe = coe
        coe[0,0] = coe[1,1] = 0
        coe[0,2] = coe[2,0] = 1/8
        coe[0,1] = 0
        coe[1,0] = nn.Parameter(torch.sin(2*np.pi*xy[...,1]))
        self.coe10 = coe[1,0]
    def __init__(self, max_dt=4e-5, mesh_size=(128,128), mesh_bound=((0,0),(1,1)), timescheme='rk2', spatialscheme='uw2'):
        super(CDE, self).__init__(max_dt=max_dt, mesh_size=mesh_size, mesh_bound=mesh_bound)
        self._timescheme = timescheme
        self._spatialscheme = spatialscheme
class Heat(_CDE, LinearTimeStepper):
    def setcoe(self, xy):
        coe = np.ndarray((3,3), dtype=object)
        self._coe = coe
        coe[0,0] = coe[1,1] = coe[0,1] = coe[1,0] = 0
        coe[0,2] = coe[2,0] = 1/8
    def __init__(self, max_dt=4e-5, mesh_size=(128,128), mesh_bound=((0,0),(1,1)), timescheme='rk2', spatialscheme='uw2'):
        super(Heat, self).__init__(max_dt=max_dt, mesh_size=mesh_size, mesh_bound=mesh_bound)
        self._timescheme = timescheme
        self._spatialscheme = spatialscheme

class _SingleCell1(_CDE):
    def setcoe(self, xy):
        coe = np.ndarray((3,3), dtype=object)
        self._coe = coe
        coe[0,0] = coe[1,1] = 0
        coe[0,2] = coe[2,0] = 1/8
        xy = xy/self.epsilon
        xy = xy%1.0
        coe[0,1] = 0
        coe[1,0] = nn.Parameter(torch.sin(2*np.pi*xy[...,1]))
        coe[1,0].data /= self.epsilon
        self.coe10 = coe[1,0]
    def __init__(self, max_dt, epsilon, cell_num, mesh_size=(256,256)):
        mesh_bound = ((0,0),(cell_num*epsilon,cell_num*epsilon))
        self.epsilon = epsilon
        self.cell_num = cell_num
        super(_SingleCell1, self).__init__(max_dt=max_dt, mesh_size=mesh_size, mesh_bound=mesh_bound)
class _SingleCell2(_CDE):
    def setcoe(self, xy):
        coe = np.ndarray((3,3), dtype=object)
        self._coe = coe
        coe[0,0] = coe[1,1] = 0
        coe[0,2] = coe[2,0] = 1/8
        xy = xy/self.epsilon
        xy = xy%1.0
        coe[0,1] = nn.Parameter(-(8*xy[...,1]-4)*torch.sin(2*np.pi*xy[...,1])-8*np.pi*(xy[...,1]-1)*xy[...,1]*torch.cos(2*np.pi*xy[...,1]))
        coe[0,1].data /= self.epsilon*2*np.pi
        coe[1,0] = nn.Parameter(-2*np.pi*torch.cos(2*np.pi*xy[...,0]))
        coe[1,0].data /= self.epsilon*2*np.pi
        self.coe01 = coe[0,1]
        self.coe10 = coe[1,0]
    def __init__(self, max_dt, epsilon, cell_num, mesh_size=(256,256)):
        mesh_bound = ((0,0),(cell_num*epsilon,cell_num*epsilon))
        self.epsilon = epsilon
        self.cell_num = cell_num
        super(_SingleCell2, self).__init__(max_dt=max_dt, mesh_size=mesh_size, mesh_bound=mesh_bound)

class SingleCell1(_SingleCell1, LinearTimeStepper):
    def __init__(self, max_dt, epsilon, cell_num, mesh_size, timescheme='rk2', spatialscheme='uw2'):
        super(SingleCell1, self).__init__(max_dt=max_dt, epsilon=epsilon, cell_num=cell_num, mesh_size=mesh_size)
        self._timescheme = timescheme
        self._spatialscheme = spatialscheme
class SingleCell2(_SingleCell2, LinearTimeStepper):
    def __init__(self, max_dt, epsilon, cell_num, mesh_size, timescheme='rk2', spatialscheme='uw2'):
        super(SingleCell2, self).__init__(max_dt=max_dt, epsilon=epsilon, cell_num=cell_num, mesh_size=mesh_size)
        self._timescheme = timescheme
        self._spatialscheme = spatialscheme
class SingleCell1Spect(_SingleCell1, LinearSpectStepper):
    def __init__(self, max_dt, epsilon, cell_num, mesh_size, timescheme='rk2'):
        super(SingleCell1Spect, self).__init__(max_dt=max_dt, epsilon=epsilon, cell_num=cell_num, mesh_size=mesh_size)
        self._timescheme = timescheme
class SingleCell2Spect(_SingleCell2, LinearSpectStepper):
    def __init__(self, max_dt, epsilon, cell_num, mesh_size, timescheme='rk2'):
        super(SingleCell2Spect, self).__init__(max_dt=max_dt, epsilon=epsilon, cell_num=cell_num, mesh_size=mesh_size)
        self._timescheme = timescheme

def test_SingleCell(cell_num=4, example=1, max_dt=1e-9, T=2e-7, epsilon=1/512):
    import aTEAM.pdetools as pdetools
    import aTEAM.pdetools.example.cde2d as cde2d
    import torch
    import matplotlib.pyplot as plt
    import time
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = None
    if example == 1:
        SingleCell = cde2d.SingleCell1
        SingleCellSpect = cde2d.SingleCell1Spect
    else:
        SingleCell = cde2d.SingleCell2
        SingleCellSpect = cde2d.SingleCell2Spect
    batch_size = 2
    h = plt.figure()
    a0 = h.add_subplot(121)
    a1 = h.add_subplot(122)
    linpde0 = SingleCell(max_dt=max_dt, epsilon=epsilon, cell_num=cell_num, mesh_size=(32*cell_num,32*cell_num))
    linpde1 = SingleCellSpect(max_dt=max_dt, epsilon=epsilon, cell_num=cell_num, mesh_size=(32*cell_num,32*cell_num))
    linpde0.to(device=device)
    linpde1.to(device=device)
    x0 = pdetools.init.initgen(mesh_size=linpde0.mesh_size, freq=1, device=device, batch_size=batch_size)
    x1 = x0
    for i in range(1,21):
        startt = time.time()
        with torch.no_grad():
            x0 = linpde0.predict(x0, T=T)
            x1 = linpde1.predict(x1, T=T)
        stopt = time.time()
        print('elapsed-time={:.1f}, sup(|x0-x1|)={:.2f}'.format(stopt-startt, (x0-x1).abs().max().item()))
        a0.clear()
        a1.clear()
        xplot0 = x0 if batch_size == 1 else x0[0]
        xplot1 = x1 if batch_size == 1 else x1[0]
        b0 = a0.imshow(xplot0, cmap='jet')
        b1 = a1.imshow(xplot1, cmap='jet')
        a0.set_title('t={:.1e},max={:.2f},min={:.2f}'.format(i*T,x0.max(),x0.min()))
        a1.set_title('t={:.1e},max={:.2f},min={:.2f}'.format(i*T,x1.max(),x1.min()))
        if i > 1:
            c0.remove()
            c1.remove()
        c0 = h.colorbar(b0, ax=a0)
        c1 = h.colorbar(b1, ax=a1)
        plt.pause(1e-3)
def test_CDE():
    import aTEAM.pdetools as pdetools
    import aTEAM.pdetools.example.cde2d as cde2d
    import torch
    import matplotlib.pyplot as plt
    import time
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = None
    T = 1e-2
    batch_size = 1
    h = plt.figure()
    a0 = h.add_subplot(121)
    a1 = h.add_subplot(122)
    linpde0 = cde2d.CDE()
    linpde1 = cde2d.Heat()
    linpde0.to(device=device)
    linpde1.to(device=device)
    linpde0.coe[0,2] = 0
    linpde0.coe[2,0] = 0
    linpde1.coe[0,2] = 1/4
    linpde1.coe[2,0] = 1/16
    init = pdetools.init.initgen(mesh_size=linpde0.mesh_size, freq=1, device=device, batch_size=batch_size)
    x0 = init
    x1 = init
    for i in range(1,21):
        startt = time.time()
        with torch.no_grad():
            x0 = linpde0.predict(x0, T=T)
            x1 = linpde1.predict(x1, T=T)
        stopt = time.time()
        print('eplapsed-time={:.1f}'.format(stopt-startt))
        a0.clear()
        a1.clear()
        xplot0 = x0 if batch_size == 1 else x0[0]
        xplot1 = x1 if batch_size == 1 else x1[0]
        b0 = a0.imshow(xplot0, cmap='jet')
        b1 = a1.imshow(xplot1, cmap='jet')
        a0.set_title('t={:.1e},max={:.2f},min={:.2f}'.format(i*T,x0.max(),x0.min()))
        a1.set_title('t={:.1e},max={:.2f},min={:.2f}'.format(i*T,x1.max(),x1.min()))
        if i > 1:
            c0.remove()
            c1.remove()
        c0 = h.colorbar(b0, ax=a0)
        c1 = h.colorbar(b1, ax=a1)
        plt.pause(1e-3)
