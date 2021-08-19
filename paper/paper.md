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
approximate Laplace interpolation [@Press1992] when no parameters are given, and
when parameters $m\in\mathbb{Z}$ and $\epsilon > 0$ are set, the interpolant
approximates a Mat\`ern kernel, of which polyharmonic splines are a special
case. 

# Mathematical Background

[@Fasshauer2012] uses the equivalent Green's function representation to 
express the dense spline matrix as a relatively sparse one. The resulting matrix solve
requires a fraction of the time required to solve the exact problem. The method hinges 
on the use of multidimensional discrete Laplacian matrices on a regular grid.ents a 

# Statement of Need

While there exist numerous implementations of interpolation routines that
fill missing data points on arbitrary grids, these are largely restricted to one
and two dimensions.

Existing, related software includes
[Laplacians.jl](https://github.com/danspielman/Laplacians.jl), whose function
`harmonic_interp` is similar to our vanilla implementation. 

# References
