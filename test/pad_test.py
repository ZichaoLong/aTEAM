#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
from numpy import *
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import aTEAM.nn.functional
#%%
npa = random.randn(1,10,7)
nppad = [[0,0],[9,3],[1,0]]
paddings = [1,0,9,3]
a = torch.from_numpy(npa)
a.requires_grad = True
b = aTEAM.nn.functional.periodicpad(a,paddings)
npb = np.pad(npa,nppad,mode='wrap')
print(linalg.norm(b.data.numpy()-npb))
#%%

