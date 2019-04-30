import numpy as np
import torch
from .upwind import *
from .spectral import *

__all__ = ['PDEStepper', 'TimeStepper', 'LinearTimeStepper', 'SpectStepper', 'LinearSpectStepper']

class PDEStepper(object):
    def step(self, init, dt, **kw):
        raise NotImplementedError
    def predict(self, init, T, **kw):
        if not hasattr(self, 'max_dt'):
            return self.step(init, T, **kw)
        else:
            n = int(np.ceil(T/self.max_dt))
            if n == 0:
                return init
            dt = T/n
            u = init
            for i in range(n):
                u = self.step(u, dt, **kw)
            return u

class TimeStepper(PDEStepper):
    @property
    def dim(self):
        raise NotImplementedError
    @property
    def timescheme(self):
        raise NotImplementedError
    @property
    def RightHandItems(self, u, **kw):
        raise NotImplementedError
    def _step_(self, u, dt, **kw):
        if self.timescheme == 'rk4': # classical 4-stage 4th-order Runge–Kutta method
            K1 = self.RightHandItems(u, **kw)
            K2 = self.RightHandItems(u+dt/2*K1, **kw)
            K3 = self.RightHandItems(u+dt/2*K2, **kw)
            K4 = self.RightHandItems(u+dt*K3, **kw)
            rhi = dt/6*(K1+2*K2+2*K3+K4)
        elif self.timescheme == 'rk2': # 2-stage 2nd-order Runge–Kutta method
            K1 = self.RightHandItems(u, **kw)
            K2 = self.RightHandItems(u+dt*K1, **kw)
            rhi = dt/2*(K1+K2)
        else: # forward Euler
            rhi = dt*self.RightHandItems(u, **kw)
        return rhi
    def step(self, init, dt, **kw):
        u = init
        u = u+self._step_(init, dt, **kw)
        return u

class LinearTimeStepper(TimeStepper):
    """
    Solver of second-order linear partial differential equations
    """
    @property
    def coe(self):
        raise NotImplementedError
    def RightHandItems(self, u):
        if self.dim == 1:
            return UpWind1dRHI(self.dx, self.coe, u, self.spatialscheme)
        elif self.dim == 2:
            return UpWind2dRHI(self.dx, self.coe, u, self.spatialscheme)
        elif self.dim == 3:
            return UpWind3dRHI(self.dx, self.coe, u, self.spatialscheme)

class SpectStepper(PDEStepper):
    """
    Pseudo-spectral method solver
    """
    @property
    def dim(self):
        raise NotImplementedError
    @property
    def timescheme(self):
        raise NotImplementedError
    @property
    def RightHandItemsSpect(self, u_spect, **kw):
        raise NotImplementedError
    def _step_(self, u_spect, dt, **kw):
        if self.timescheme == 'rk4': # classical 4-stage 4th-order Runge–Kutta method
            K1 = self.RightHandItemsSpect(u_spect, **kw)
            K2 = self.RightHandItemsSpect(u_spect+dt/2*K1, **kw)
            K3 = self.RightHandItemsSpect(u_spect+dt/2*K2, **kw)
            K4 = self.RightHandItemsSpect(u_spect+dt*K3, **kw)
            rhi_spect = dt/6*(K1+2*K2+2*K3+K4)
        elif self.timescheme == 'rk2': # 2-stage 2nd-order Runge–Kutta method
            K1 = self.RightHandItemsSpect(u_spect, **kw)
            K2 = self.RightHandItemsSpect(u_spect+dt*K1, **kw)
            rhi_spect = dt/2*(K1+K2)
        else: # forward Euler
            rhi_spect = dt*self.RightHandItemsSpect(u_spect, **kw)
        return rhi_spect
    def step(self, init, dt, **kw):
        init = torch.stack([init, torch.zeros_like(init)], dim=-1)
        u_spect = torch.ifft(init, signal_ndim=self.dim)
        u_spect = u_spect+self._step_(u_spect, dt, **kw)
        # print(np.abs(fft.fftn(u_spect).imag).max())
        u = torch.fft(u_spect, signal_ndim=self.dim)
        u = u[...,0]
        return u

class LinearSpectStepper(SpectStepper):
    """
    Solver of second-order linear partial differential equations
    """
    def RightHandItemsSpect(self, u_spect, **kw):
        if self.dim != 2:
            raise NotImplementedError
        coe = self.coe
        rhi = 0
        for k in range(3):
            for j in range(k+1):
                if isinstance(coe[j,k-j], (int,float)):
                    if coe[j,k-j] == 0:
                        continue
                u_diff = spect_diff(u_spect, signal_ndim=2, order=(j,k-j), mesh_bound=self.mesh_bound)
                u_diff = spect2time(u_diff, signal_ndim=2)
                if k == 1:
                    tmp = -coe[j,k-j]
                else:
                    tmp = coe[j,k-j]
                if isinstance(tmp, torch.Tensor):
                    rhi = rhi+SpectMul(tmp, u_diff, signal_ndim=2)
                else:
                    rhi = rhi+u_diff*tmp
        return time2spect(rhi, signal_ndim=2)
