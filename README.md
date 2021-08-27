# Fast Interpolation for Volume Datasets

[![Build Status](https://github.com/vishwas1984/LaplaceInterpolation.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/vishwas1984/LaplaceInterpolation.jl/actions)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://vishwas1984.github.io/LaplaceInterpolation.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://vishwas1984.github.io/LaplaceInterpolation.jl/dev)

This code performs Laplace and Matern interpolation where missing data are on a one, two, or three
dimensional grid. Matern
kernels generalize the radial basis function approach to interpolation, but
interpolation using these kernels 
involves systems of equations that are dense. By using the Green's function
representation, and substituting the finite-difference operator, we replace the dense operator with a sparse one
and thus obtain an approximation to the kernel.

TL;DR: Substituting a discrete Laplace approximation to Matern kernel reduces the computational complexity of gridded interpolation from $M^3$ (if $M$ is the product of the size of the data in each dimension of a hyperrectangular d-dimensional dataset) to $M$, and hence proceeds extremely fast. 

# Installation

This package is unregistered, so please install using

```
pkg> add https://github.com/vishwas1984/LaplaceInterpolation.jl
```

# Notebooks
Jupyter Notebooks which illustrate the speed and accuracy of the approximation
are located in the `/Notebooks` directory.

To run the examples yourself, clone the repo, navigate to the Notebooks
directory, start julia and use
```
pkg> activate .
pkg> instantiate
julia> include("run_notebooks.jl") 
```
which will start a jupyter notebook for you, with all relevant dependencies.

# Documentation 

Forthcoming. Please refer to the Examples notebooks for useage. Github displays notebooks with output, if you do not wish to run them yourself.

# Sample results

Below we show an example of a three dimensional x-ray scattering experiment on
a crystalline structure in which peaks at integer locations are removed from the
dataset and interpolated. The plot shows a one-dimensional cut of the 3D data along
the h-axis. The image on the left shows the data with and without interpolation (the
original data is in red, Green and Orange respectively show Laplace and Matern
interpolated data). The right hand side image is a blown-up version of that on the left,
using a linear scale in the y-axis.


Bragg Peaks                | Matern and Laplace Interpolation 
:-------------------------:|:--------------------------------:
![](docs/BraggPeaks.png)  |  ![](docs/Punch_Fill.png)

For large three-dimensional datasets such as this one, an RBF kernel interpolation
fails on a laptop computer with contemporary (2021) architecture. The gridded interpolation
approach we give here can tackle the above $9,000,000 \times 9,000,000$ problem on
the same computer with ease.

# Similar Packages
```Laplacians.jl``` authored by Dan Spielman.

# Funding
This material is based upon work supported by the U.S. Department of Energy,
Office of Science, Office of Basic Energy Sciences.

