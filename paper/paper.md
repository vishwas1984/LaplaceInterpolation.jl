---
title: 'LaplaceInterpolation.jl: A Julia package for fast interpolation on a grid'
tags:
  - Julia
  - statistics
  - spatial statistics
  - space-time processes
authors:
  - name: Vishwas Rao
    affiliation: "1"
  - name: Charlotte L. Haley
    orcid: 0000-0003-3996-773X
    affiliation: "1"
    email: "haley@anl.gov" 
affiliations:
 - name: Argonne National Laboratory
   index: 1
date: 28 Jun, 2021
bibliography: paper.bib

---

# Summary

We introduce a linear-time algorithm for interpolation on a regular
multidimensional grid, implemented in the Julia language. The algorithm is an
approximate Laplace interpolation [@press1992] when no parameters are given, and
when parameters $m\in\mathbb{Z}$ and $\epsilon > 0$ are set, the interpolant
approximates a Mat\`ern kernel, of which radial basis functions and polyharmonic
splines are a special case. 

# Mathematical Background

[@fasshauer2012green] uses the equivalent Green's function representation to 
express the dense spline matrix as a relatively sparse one. The resulting matrix solve
requires a fraction of the time required to solve the exact problem. The method hinges 
on the use of multidimensional discrete Laplacian matrices on a regular grid. 

# Statement of Need

While there exist numerous implementations of interpolation routines that fill
missing data points on arbitrary grids, these are (i) largely restricted to one
and two dimensions (ii) slow to run. The implementation we propose is
dimension-agnostic, based on a linear-time algorithm, and implements an
approximate Matern kernel interpolation (of which thin plate splines,
polyharmonic splines, and radial basis functions are a special case.)  

# Why is it so fast?

This is because the problem largely boils down to the solution of $Ax = b$
[@mainberger2011optimising] where the square matrix $A$'s size is the product of
the number of points in each of the dimensions, and is dense.  For the special
case where the data points are on a regular grid, and the Matern kernel
interpolant is used, a remarkable simplification occurs, in which a discrete
approximation to the Green's function for the operator results in an interpolant
having sparse matrix representation.  

# Other software for interpolation

Existing, related software includes, as of the time of this writing

## Julia 

* [Interpolations.jl](https://github.com/JuliaMath/Interpolations.jl) does
  B-splines and Lanczos interpolation, and has support for irregular grids.
* [Dieerckx.jl](https://github.com/kbarbary/Dierckx.jl) a julia-wrapped Fortran
  package for 1-D and 2-D splins.
* [GridInterpolations.jl](https://github.com/sisl/GridInterpolations.jl) 
* [Laplacians.jl](https://github.com/danspielman/Laplacians.jl), whose function
`harmonic_interp` is similar to our vanilla implementation. 

## Python

* [astropy.convolve](https://docs.astropy.org/en/stable/api/astropy.convolution.convolve.html) will interpolate gridded data by rescaling a convoution kernel when it encounters missing values.
* [scipy.interpolate.RBF](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html)

## Other


# References
