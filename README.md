# HaarMaternKernel
This program offers the possibility to compute Gaussian process regression with Matern Kernel in the Haar basis. This code is based on [GPy](https://sheffieldml.github.io/GPy/).

## To use this code
The code of the Covariance kernel is available in the file [KernelHaarMatern.py](https://github.com/ehbihenscoding/HaarMaternKernel/blob/main/KernelHaarMatern.py). 
However, it is recommended that the regression be performed with file [GPregressionWaveketCov.py](https://github.com/ehbihenscoding/HaarMaternKernel/blob/main/GPregressionWaveletCov.py). In particular because the covariance structure is defined for a given input structure.
It is possible to change the data by replacing the data t, Nt, xH, yH and Exact.

## Warning
The code can give good results in a very small data case but is not yet operational 