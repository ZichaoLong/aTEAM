#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
from numpy import *
import numpy as np
import torch
import aTEAM.nn.functional as aF
#%%
a = np.arange(10)
a = a[:,None]+a[None,:]
b = torch.from_numpy(a)

print(np.roll(a, shift=[0,1],axis=[1,0])-aF.roll(b, shift=[0,1],axis=[1,0]).data.numpy())
print(np.roll(a, shift=[2,1])-aF.roll(b, shift=[2,1]).data.numpy())
print(np.roll(a, shift=[2,-1], axis=[0,1])-aF.roll(b, shift=[2,-1], axis=[0,1]).data.numpy())
#%%
