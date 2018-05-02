import torch
from functools import reduce

__all__ = ['tensordot','periodicpad']

def periodicpad(inputs, pad):
    """
    'periodic' pad, similar to torch.nn.functional.pad 
    """
    n = inputs.dim()
    inputs = inputs.permute(*list(range(n-1,-1,-1)))
    pad = iter(pad)
    i = 0
    indx = []
    for a in pad:
        b = next(pad)
        assert a<inputs.size()[i] and b<inputs.size()[i]
        permute = list(range(n))
        permute[i] = 0
        permute[0] = i
        inputs = inputs.permute(*permute)
        inputlist = [inputs,]
        if a > 0:
            inputlist = [inputs[slice(-a,None)],inputs]
        if b > 0:
            inputlist = inputlist+[inputs[slice(0,b)],]
        if a+b > 0:
            inputs = torch.cat(inputlist,dim=0)
        inputs = inputs.permute(*permute)
        i += 1
    inputs = inputs.permute(*list(range(n-1,-1,-1)))
    return inputs

def tensordot(a,b,dim):
    """
    tensordot in PyTorch, see numpy.tensordot?
    """
    l = lambda x,y:x*y
    if isinstance(dim,int):
        a = a.contiguous()
        b = b.contiguous()
        sizea = a.size()
        sizeb = b.size()
        sizea0 = sizea[:-dim]
        sizea1 = sizea[-dim:]
        sizeb0 = sizeb[:dim]
        sizeb1 = sizeb[dim:]
        N = reduce(l, sizea1, 1)
        assert reduce(l, sizeb0, 1) == N
    else:
        adims = dim[0]
        bdims = dim[1]
        adims = [adims,] if isinstance(adims, int) else adims
        bdims = [bdims,] if isinstance(bdims, int) else bdims
        adims_ = set(range(a.dim())).difference(set(adims))
        adims_ = list(adims_)
        adims_.sort()
        perma = adims_+adims
        bdims_ = set(range(b.dim())).difference(set(bdims))
        bdims_ = list(bdims_)
        bdims_.sort()
        permb = bdims+bdims_
        a = a.permute(*perma).contiguous()
        b = b.permute(*permb).contiguous()

        sizea = a.size()
        sizeb = b.size()
        sizea0 = sizea[:-len(adims)]
        sizea1 = sizea[-len(adims):]
        sizeb0 = sizeb[:len(bdims)]
        sizeb1 = sizeb[len(bdims):]
        N = reduce(l, sizea1, 1)
        assert reduce(l, sizeb0, 1) == N
    a = a.view([-1,N])
    b = b.view([N,-1])
    c = a@b
    return c.view(sizea0+sizeb1)

