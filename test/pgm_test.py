"""ParamGroupsManager example"""
#%%
import torch
import torch.nn as nn
from aTEAM.optim import ParamGroupsManager

class Penalty(nn.Module):
    def __init__(self,n,alpha=1e-5):
        super(Penalty,self).__init__()
        m = n//2
        x1 = torch.arange(1,m+1,dtype=torch.float)
        x2 = torch.arange(m+1,n+1,dtype=torch.float)
        self.x1 = nn.Parameter(x1); self.x2 = nn.Parameter(x2)
        self.n = n; self.alpha = alpha
    def forward(self):
        x = torch.cat([self.x1,self.x2],0)
        return self.alpha*((x-1)**2).sum()+((x**2).sum()-0.25)**2
penalty = Penalty(4,1e-5)
# Each of the following case is OK, 'lr'='learning_rate'
# pgm = ParamGroupsManager(params=penalty.parameters(),
#         defaults={'lr':0.1,'scale':10})
# pgm = ParamGroupsManager(params=penalty.named_parameters(),
#         defaults={'lr':0.1,'scale':10})
pgm = ParamGroupsManager(params=[
        {'params':[penalty.x1,]},{'params':{'x2':penalty.x2},'lr':0.2}
        ],defaults={'lr':0.1,'scale':10})
# show what ParamGroupsManager does:
print("pgm.param_groups")
print(pgm.param_groups)
print("\npgm.params")
print(list(pgm.params))
print("\npgm.named_params")
print(list(pgm.named_params))
print("\npgm.params_with_info 'scale' and 'lr' ")
print(list(pgm.params_with_info('scale','lr')))
#%%


