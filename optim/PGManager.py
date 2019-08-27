"""torch parameter groups manager"""
import torch
from collections import OrderedDict,Iterator

__all__ = ['ParamGroupsManager',]

class ParamGroupsManager(object):
    """
    `ParamGroupsManager` is a class for managing torch parameter groups.
    self.defaults = dict(options0=v0,options1=v1,...), 
    self.param_groups = [param_group0,param_group1,...], where 
        param_group = dict(
                           params=OrderedDict([(name0,tensor0),...]),
                           # options 
                           key0=..., # with default value self.defaults[key0]
                           key1=..., #        ...         self.defaults[key1]
                           key2=..., #        ...         self.defaults[key2]
                           ...)
    `ParamGroupsManager` provides several interfaces to access parameters:
        self.params,self.named_params,self.self.params_with_info,
    Please refer to the docstrings of these properties.

    .. note:: 
    :class:`ParamGroupsManager` is similar to :class:`Optimizer.param_groups`. 
    The main difference between them is how to store parameters:
        for param_group in ParamGroupsManager.param_groups:
            param_group['params'] = an OrderedDict of named_parameters
        for param_group in :class:`torch.Optimizer.param_groups`:
            param_group['params'] = a list of parameters

    Arguments:
        params (iterable): params specifies what tensors should be managed. 
            params will be convert to self.param_groups. 
            Each of the following cases for params is OK, 
            1). params = [tensor0, tensor1, tensor2, ...]
                -> self.param_groups = [
                        {'params':OrderedDict(enumerate(params)),
                        ...options...}
                        ]
            2). params = dict(name0=tensor0, name1=tensor1, ...)
                -> self.param_groups = [
                        {'params':OrderedDict(params),
                        ...options...}
                        ]
            3). params = [
                    {'params':[tensor00,tensor01,...], key0:v00,...},
                    {'params':[tensor10,tensor11,...], key0:v01,...},
                    ...
                    ]
                -> self.param_groups = [
                    {'params':OrderedDict(enumerate(params[0]['params'])), 
                        key0:v00,...},
                    {'params':OrderedDict(enumerate(params[1]['params'])), 
                        key0:v01,...},
                    ...
                    ]
            4). params = [
                    {'params':{name00:tensor00,...}, key0:v00,...},
                    {'params':{name10:tensor10,...}, key0:v01,...},
                    ...
                    ]
                -> self.param_groups = [
                    {'params':OrderedDict(params[0]['params']), 
                        key0:v00,...},
                    {'params':OrderedDict(params[1]['params']), 
                        key0:v01,...},
                    ...
                    ]
            self.param_groups will be initialized in self.__init__, if you 
            want to add param_groups to self.param_groups after initialization,
            you can use 
                self.add_param_group(param_group), 
            where param_group should be a list or dict of tensors (see 1),2)) 
            or is already a param_group:
                    {'params':[tensor00,tensor01,...], key0:v00,...} or 
                    {'params':{name00:tensor00,...}, key0:v00,...},
        defaults (dict): default options for parameter groups. Different from 
            parameters(i.e. params). Options can also be set in
            augument `params`.
    Example:
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
        pgm = ParamGroupsManager(params=penalty.parameters(),
                defaults={'lr':0.1,'scale':10})
        pgm = ParamGroupsManager(params=penalty.named_parameters(),
                defaults={'lr':0.1,'scale':10})
        pgm = ParamGroupsManager(params=[
                {'params':[penalty.x1,]},{'params':{'x2':penalty.x2},'lr':0.2}
                ],defaults={'lr':0.1,'scale':10})
        # show what ParamGroupsManager does:
        print("pgm.param_groups"); print(pgm.param_groups)
        print("\npgm.params"); print(list(pgm.params))
        print("\npgm.named_params"); print(list(pgm.named_params))
        print("\npgm.params_with_info 'scale' and 'lr' ")
        print(list(pgm.params_with_info('scale','lr')))
    """
    def __init__(self, params, defaults):
        self.defaults = defaults

        # set param_groups
        self.param_groups = []
        if isinstance(params, Iterator):
            params = list(params)
        _is_params,params_tmp = ParamGroupsManager.is_params(params)
        if _is_params:
            param_group = dict(params=params_tmp)
            self.add_param_group(param_group)
        else:
            for param_group in params:
                if isinstance(param_group, dict) and 'params' in param_group:
                    pg = self._copy_options(param_group)
                    _is_params,params_tmp = \
                            ParamGroupsManager.is_params(param_group['params'])
                    assert _is_params, \
                            "param_group['params'] is expected to pass \
                            ParamGroupsManager.is_params, \
                            see ParamGroupsManager.is_params?"
                    pg['params'] = params_tmp
                else:
                    raise ValueError("param_group is expceted to be a dict "
                            "with key 'params'")
                self.add_param_group(pg)

    # is_params, is_param_group
    @staticmethod
    def _copy_options(param_group):
        p = {}
        for k,v in param_group.items():
            if k != 'params':
                p[k] = v
        return p
    @staticmethod
    def _pack_params(p):
        if isinstance(p,Iterator):
            p = list(p)
        if isinstance(next(iter(p)), torch.Tensor):
            p = enumerate(p)
        p = OrderedDict(p)
        return p
    @staticmethod
    def is_params(params):
        """
        Verify whether params is an iterable of parmeters.
        An iterable of (name, :class:`torch.Tensor`) pairs or :class:`torch.Tensor` s 
        will pass this judgement function. So does named Variables dict.

        Example:
            >>> model = nn.Linear(3,2)
            >>> ParamGroupsManager.is_params(model.parameters())
            (True,OrderedDict([(0,...),(1,...)]))
            >>> ParamGroupsManager.is_params(model.named_parameters())
            (True,OrderedDict([('weight',...),('bias',...)]))
            >>> ParamGroupsManager.is_params(dict(model.named_parameters()))
            (True,OrderedDict([('weight',...),('bias',...)]))
            >>> ParamGroupsManager.is_params([model.weight,])
            (True,OrderedDict([(0,...),]))
            >>> ParamGroupsManager.is_params([model.weight.data,])
            (False,OrderedDict([(0,...),(1,...)])) # split model.weight.data
        """
        try:
            if isinstance(params, torch.Tensor):
                # in some case, people unconsciously pass a tensor in, 
                # which is also a iterable of tensor when size>1.
                params = [params,]
            if isinstance(params, Iterator):
                # an Iterator can use only once, 
                # we should at first convert it to a list.
                params = list(params)
            assert len(list(params))>0, "got empty params"
            if not isinstance(params, dict):
                params = list(params)
                if isinstance(params[0], torch.Tensor):
                    b = all(map(lambda v:isinstance(v, torch.Tensor), params))
                else: # expect to be a list of (name, :class:`torch.Tensor`) pairs
                    params = dict(params)
            if isinstance(params, dict):
                b = all(map(lambda v:isinstance(v[1], torch.Tensor), params.items()))
            assert b
            return True,ParamGroupsManager._pack_params(params)
        except:
            return False,params
    @staticmethod
    def is_param_group(param_group):
        """See the code."""
        if isinstance(param_group, dict) and ('params' in param_group):
            _is_params,params_tmp = \
                    ParamGroupsManager.is_params(param_group['params'])
            if _is_params:
                pg = ParamGroupsManager._copy_options(param_group)
                pg['params'] = params_tmp
                return True,pg
        return False,None
    # add_param_group
    def add_param_group(self, param_group):
        """Add a param group to self.param_groups

        This can be useful when you want to add optimization parameters 
        during training.

        Arguments:
            param_group (dict or params): Specifies what Variables should be 
                added to be managed.
                assert  
                    ParamGroupsManager.is_params(param_group)[0] or 
                    ParamGroupsManager.is_param_group(param_group)[0]
        """
        _is_params,params_tmp = ParamGroupsManager.is_params(param_group)
        _is_param_group,param_group_tmp = \
                ParamGroupsManager.is_param_group(param_group)
        assert _is_params or _is_param_group, \
                "invalid param_group, see \
                ParamGroupsManager.is_params?,\
                ParamGroupsManager.is_param_group?"
        if _is_params:
            param_group_tmp = dict(params=params_tmp)

        for k,v in self.defaults.items():
            param_group_tmp.setdefault(k, v)

        # Verify whether there are duplicate parameters.
        params_candidate = list(map(lambda x:id(x[1]), \
                param_group_tmp['params'].items()))
        assert len(set(params_candidate)) == len(params_candidate), \
                'parameter in param_group should be unique'
        assert set(params_candidate).isdisjoint(set(map(id, self.params))), \
                'duplicate parameter in param_group and self.params'

        self.param_groups.append(param_group_tmp)
        return None

    # params iterator of ParamGroupsManager
    @property
    def params(self):
        for param_group in self.param_groups:
            for _,v in param_group['params'].items():
                yield v
    @property
    def named_params(self):
        for param_group in self.param_groups:
            for name,v in param_group['params'].items():
                yield name,v
    def params_with_info(self, *keys):
        for param_group in self.param_groups:
            value = []
            for k in keys:
                value.append(param_group[k])
            for _,v in param_group['params'].items():
                yield value+[v,]

    # zero_grad
    def zero_grad(self):
        """
        Clears the gradients of all managed :class:`torch.Tensor` s.
        The code is almost simply copied from torch.optim.optimizer.
        """
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

