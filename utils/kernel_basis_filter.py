#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
__all__ = ['circshift', 'dx_filter_coe', 'dy_filter_coe','diff_monomial_coe',
        'wrap_filter2d', 'dx_filter', 'dy_filter', 
        'single_moment', 'switch_moment_filter', 'total_moment', 
        'psf2otf', 'coe2hat', 'diff_op_default_coe']
from numpy import *
from numpy.fft import *
from numpy.linalg import *
from scipy.signal import correlate,correlate2d,convolve2d
from scipy.special import factorial
from functools import reduce
#%%
def circshift(ker, shape, *, cval=0):
    # [0,1,...,M]中的[(M+1)/2]移动到原点
    s0 = ker.shape
    pad_width = []
    for i in range(len(shape)):
        assert shape[i] > s0[i]
        pad_width.append([0,shape[i]-s0[i]])
    z = pad(ker, pad_width, mode='constant', constant_values=cval)
    for i in range(ker.ndim):
        z = roll(z, -(s0[i]//2), axis=i)
    return z
def wrap_filter2d(ker, *, method='origin', **kw):
    """
    Args:
        ker: correlate kernel
        method: fft,origin
    Return:
        a callable correlate filter, the correlate kernel given by ker. 
        This filter take an 2d ndarray and padding mode and boundary condition as args.
            mode: same, full, valid
            boundary: fill, wrap, symm
    Usage:
        ker = array([[0,1,0],[1,-4,1],[0,1,0]])
        f = wrap_filter2d(ker, method=origin)
        a = random.randn(10,10)
        b = f(a, mode='same', boundary='wrap')
    """
    def f(u, mode='same', boundary='wrap'):
        v = zeros(u.shape)
        if u.ndim == 2:
            v = correlate2d(u, ker, mode=mode, boundary=boundary)
            return v
        for i in range(u.shape[0]):
            v[i,:,:] = correlate2d(u[i,:,:], ker, mode=mode, boundary=boundary)
        return v
    def g(u):
        v = zeros(u.shape)
        ker_h = coe2hat(ker, u.shape)
        if u.ndim == 2:
            v = (ifft2(fft2(u)*ker_h)).real
            return v
        ker_h = reshape(ker_h, [1,*ker_h.shape])
        v = (ifft2(fft2(u)*ker_h)).real
    if method == 'origin':
        return f
    else:
        return g
def dx_filter_coe(ver=0):
    """
    差分算子discrete filter
    用于correlate而非convolve
    """
    ##ver-1
    if ver == 1:
        l = sqrt(2)-1
        a = zeros([3,3])
        a[1,0] = -l/2
        a[1,2] = l/2
        a[[0,2],0] = -(1-l)/4
        a[[0,2],2] = (1-l)/4
    elif ver == 2:
        a = zeros([3,3])
        a[1,0] = -0.5
        a[1,2] = 0.5
    elif ver == 3:
        a = zeros([3,3])
        a[1,1] = -1
        a[1,2] = 1
    elif ver == 4:
        a = zeros([3,3])
        a[1,0] = -1
        a[1,1] = 1
    else:
    ##ver-0
        a = zeros((1,2))
        a[0,0] = -1
        a[0,1] = 1
    return a
def dy_filter_coe(ver=0):
    """
    return dx_filter_coe(ver).transpose()
    """
    return dx_filter_coe(ver).transpose()
def diff_monomial_coe(x_order=0, y_order=0, x_vers=None, y_vers=None, shape=None):
    if x_vers is None:
        k = x_order//2
        l = x_order%2
        x_vers = [3,4]*k+[2]*l
    if y_vers is None:
        k = y_order//2
        l = y_order%2
        y_vers = [3,4]*k+[2]*l
    ker = ones([1,1])
    for v in x_vers:
        ker = convolve2d(ker, dx_filter_coe(v)[::-1,::-1])
    for v in y_vers:
        ker = convolve2d(ker, dy_filter_coe(v)[::-1,::-1])
    ker = ker[::-1,::-1]
    n,m = nonzero(ker)
    lb_row, ub_row = min(n),max(n)
    rowindx = min(lb_row, ker.shape[0]-ub_row-1)
    lb_col, ub_col = min(m),max(m)
    colindx = min(lb_col, ker.shape[1]-ub_col-1)
    ker = ker[rowindx:ker.shape[0]-rowindx,colindx:ker.shape[1]-colindx]
    if not shape is None:
        pady = shape[0]-ker.shape[0]
        padx = shape[1]-ker.shape[1]
        assert padx>=0 and pady>=0
        ker = pad(
                ker, 
                pad_width=[[pady-pady//2,pady//2],[padx-padx//2,padx//2]],
                mode='constant'
                )
    return ker
def diff_op_default_coe(shape, op='laplace'):
    assert op in ['laplace','dx','dy','grad','div']
    shape = [shape[0], shape[1]]
    if op == 'laplace':
        ker = diff_monomial_coe(shape=shape, x_order=2)+diff_monomial_coe(shape=shape, y_order=2)
    elif op == 'dx':
        ker = diff_monomial_coe(shape=shape, x_order=1)
    elif op == 'dy':
        ker = diff_monomial_coe(shape=shape, y_order=1)
    elif op == 'grad':
        ker = concatenate(
                (
                    reshape(diff_monomial_coe(shape=shape, x_order=1), shape+[1,1]),
                    reshape(diff_monomial_coe(shape=shape, y_order=1), shape+[1,1])
                    ), axis=3
                )
    elif op == 'div':
        ker = concatenate(
                (
                    reshape(diff_monomial_coe(shape=shape, x_order=1), shape+[1,1]),
                    reshape(diff_monomial_coe(shape=shape, y_order=1), shape+[1,1])
                    ), axis=2
                )
    return ker
def diff_op_default_filter(shape, op='laplace'):
    ker = diff_op_default_coe(shape, op=op)
    return wrap_filter2d(ker, method='origin')
def dx_filter(ver=0):
    return wrap_filter2d(dx_filter_coe(ver), method='origin')
def dy_filter(ver=0):
    return wrap_filter2d(dy_filter_coe(ver), method='origin')
def psf2otf(ker, shape):
    return fft2(circshift(ker, shape))
def coe2hat(ker, shape):
    return psf2otf(ker[::-1,::-1], shape)
def single_moment(ker, order=(0,0)):
    assert ker.ndim == len(order)
    l = []
    for i in range(ker.ndim):
        tmpshape = [1]*ker.ndim
        tmpshape[i] = ker.shape[i]
        l.append(reshape((arange(ker.shape[i])-ker.shape[i]//2)**order[i], tmpshape))
    return sum(reduce(multiply, [*l,ker]))/product(factorial(order))
def switch_moment_filter(shape):
    M = []
    invM = []
    assert len(shape) > 0
    for l in shape:
        M.append(zeros((l,l)))
        for i in range(l):
            M[-1][i] = ((arange(l)-l//2)**i)/factorial(i)
        invM.append(inv(M[-1]))
    def apply_axis_left_dot(x, mats):
        assert x.ndim == len(mats)
        x = x.copy()
        k = len(mats)
        for i in range(k):
            x = tensordot(mats[k-i-1], x, axes=[1,k-1])
        return x
    def apply_axis_right_dot(x, mats):
        assert x.ndim == len(mats)
        x = x.copy()
        for i in range(len(mats)):
            x = tensordot(x, mats[i], axes=[0,0])
        return x
    def m2f(m):
        return apply_axis_left_dot(m, invM)
    def f2m(f):
        return apply_axis_left_dot(f, M)
    def m2f_grad(m_grad):
        return apply_axis_right_dot(m_grad, M)
    def f2m_grad(f_grad):
        return apply_axis_right_dot(f_grad, invM)
    return m2f, f2m, m2f_grad, f2m_grad

def total_moment(ker, size=(5,5)):
    x = zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            x[i,j] = single_moment(ker, [i,j])
    return x
def test(ker):
    errs = []
    for i in range(2):
        for j in range(2):
            n = ker.shape[0]+(i+10)*9
            m = ker.shape[1]+(j+10)*9
            tmp = random.randn(n,m)
            ker_hat = coe2hat(ker, (n,m))
            tmp_filter = wrap_filter2d(ker, method='origin')
            err = mean(abs(ifft2(ker_hat*fft2(tmp))-tmp_filter(tmp)))
            errs.append(err)
    return mean(errs)
if __name__ == '__main__':
    kers = []
    for i in range(2):
        kers.append(dx_filter_coe(i))
        kers.append(dy_filter_coe(i))
    for i in range(2):
        for j in range(2):
            kers.append(random.randn(i*10+9, j*10+9))
    for ker in kers:
        print(test(ker))

#%%

