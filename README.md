# aTEAM
**A** py**T**orch **E**xtension for **A**pplied **M**athematics

This version is compatible with pytorch (1.0.1) and later. You can create a conda environment for pytorch1:
```
conda create -n torch1 python=3 jupyter
source activate torch1
conda install pytorch=1 torchvision cudatoolkit=9.2 -c pytorch
# or conda install pytorch-cpu=1 -c pytorch
```

## Some code maybe useful to you (News: add optim QuickStart)

- aTEAM.optim.NumpyFuntionInterface: This function enable us to optimize pytorch modules with external optimizer such as scipy.optimize.lbfgsb.fmin_l_bfgs_b, see test/optim_quickstart.py
- aTEAM.nn.modules.MK: [Moment matrix](https://arxiv.org/abs/1710.09668) & convolution kernel convertor: aTEAM.nn.modules.MK.M2K, aTEAM.nn.module.MK.K2M
- aTEAM.nn.modules.Interpolation: Lagrange interpolation in a n-dimensional box: aTEAM.nn.modules.Interpolation.LagrangeInterp, aTEAM.nn.modules.Interpolation.LagrangeInterpFixInputs
- aTEAM.nn.functional.utils.tensordot: It is similar to numpy.tensordot

For more usages pls refer to aTEAM/test/*.py

# PDE-Net

aTEAM is a basic library for PDE-Net & PDE-Net 2.0[(source code)](https://github.com/ZichaoLong/PDE-Net):

- [PDE-Net: Learning PDEs from Data](https://arxiv.org/abs/1710.09668)[(ICML 2018)](https://icml.cc/Conferences/2018)<br />
[Long Zichao](https://scholar.google.com/citations?user=0KXcwnkAAAAJ&hl=zh-CN), [Lu Yiping](https://web.stanford.edu/~yplu/), [Ma Xianzhong](https://www.researchgate.net/profile/Xianzhong_Ma), [Dong Bin](http://bicmr.pku.edu.cn/~dongbin)
- [PDE-Net 2.0: Learning PDEs from Data with A Numeric-Symbolic Hybrid Deep Network](https://arxiv.org/abs/1812.04426)<br />
[Long Zichao](https://scholar.google.com/citations?user=0KXcwnkAAAAJ&hl=zh-CN), [Lu Yiping](https://web.stanford.edu/~yplu/), [Dong Bin](http://bicmr.pku.edu.cn/~dongbin)

If you find this code useful for your research then please cite
```
@inproceedings{long2018pdeI,
  title={PDE-Net: Learning PDEs from Data},
  author={Long, Zichao and Lu, Yiping and Ma, Xianzhong and Dong, Bin},
  booktitle={International Conference on Machine Learning},
  pages={3214--3222},
  year={2018}
}
@article{long2018pdeII,
    title={PDE-Net 2.0: Learning PDEs from Data with A Numeric-Symbolic Hybrid Deep Network},
    author={Long, Zichao and Lu, Yiping and Dong, Bin},
    journal={arXiv preprint arXiv:1812.04426},
    year={2018}
}
```
