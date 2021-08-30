# LaplaceInterpolation.jl Documentation

This package quickly interpolates data on a grid in one and higher dimensions. 

## Installation Instructions

This package is unregistered, so please install using

```
Pkg> add https://github.com/vishwas1984/LaplaceInterpolation.jl
```

## Getting started

Suppose we need to interpolate the following image 

``` 
using LaplaceInterpolation, TestImgaes

img = Float64.(Gray.(testimage("mandrill")))
```

For illustration purposes, we'll punch a few holes and randomize some data

```
rows, columns = (256, 512)
N = rows*columns

mat = convert(Array{Float64}, imgg)[1:rows,1:columns]

N2 = Int64(round(N/2))
No_of_nodes_discarded = Int64(round(0.9*N2))

discard1 = N2 .+ randperm(N2)[1:No_of_nodes_discarded]

cent = [(150, 150), (60, 100)]
rad = 30*ones(Int64, 2)
discard2 = punch_holes_2D(cent, rad, rows, columns);

discard = vcat(discard1, discard2)
mat[discard] .= 1

heatmap(mat, title = "Image with Missing data", yflip = true, 
              c = :bone, clims = (0.0, 1.0))
```

Interpolating using the laplace and matern approximations, we get

```
restored_img_laplace = matern_2d_grid(mat, discard, 1, 0.0)
restored_img_matern = matern_2d_grid(mat, discard, 2, 0.0)
```

And plotting, we have

```
p1 = heatmap(mat, title = "Original Data", yflip = true, 
              c = :bone, clims = (0.0, 1.0))
p2 = heatmap(holeyimage1, title = "Image with Missing data", yflip = true, 
              c = :bone, clims = (0.0, 1.0))
p3 = heatmap(restored_img_laplace, title = "Laplace Interpolated Image", yflip =
true, 
              c = :bone, clims = (0.0, 1.0))
p4 = heatmap(restored_img_matern, title = "Matern, m = 2, eps = 0.0", yflip =
true, 
              c = :bone, clims = (0.0, 1.0))
plot(p1, p2, p3, p4, layout = (2, 2), legend = false, size = (900, 500))

```

![Mandrill_Random](doc)

The `Notebooks` folder contains this and other examples. 

## Mathematical Details

Radial basis functions and splines can be unified conceptually through the
notion of Green's functions and eigenfunction expansions [(Fasshauer,
2012)](https://link.springer.com/chapter/10.1007/978-1-4614-0772-0_4).  The
general multivariate Matern kernels are of the form 

$K(\mathbf x ; \mathbf z) = K_{m-d/2}(\epsilon||\mathbf x -\mathbf z ||)(ϵ||\mathbf x - \mathbf z ||)^{m-d/2}$

for $m > d/2$, where $K_ν$ is the modified Bessel function of the second kind, and can be
obtained as Green’s kernels of 

```math 
L = (ϵ^2I-Δ)^m 
```

where $Δ$ denotes the Laplacian operator in $d$ dimensions. Polyharmonic
splines, including thin plate splines, are a special case of the above, and this
class includes the thin plate splines. 

The discrete gridded interpolation seeks to find an interpolation $u (\mathbf x
)$ that satisfies the differential operator in $d$ dimensions on the nodes
$\mathbf x_i$ where there is no data and equals $y_i$ everywhere else.
Discretely, one solves the matrix problem

```math 
\mathbf C  (\mathbf u  - \mathbf y ) - (1 - \mathbf C ) L \mathbf u  = 0 
```

where $\mathbf{y}$ contains the $y_i$'s and placeholders where there is no data, $L$
denotes the discrete matrix operator and $C$ is a diagonal matrix that indicates 
whether node $\mathbf x_i$ is observed. 

In $d-$ dimensions the matrix $A^{(d)}$ of size $M \times M$ expands the first
order finite difference curvature and its $(i,j)$th entry is -1 when node j is
in the set of neighbors of the node $\mathbf x_i$, and has the number of such neighbors on the diagonal. 
Note that if node $i$ is a boundary node, the $i$-th row of $A^{(d)}$ has
$-1$s in the neighboring node spots and the number of such nodes on the
diagonal. In general, the rows of $A^{(d)}$ sum to zero. 

Denote by $L = A^{(d)}$ the discrete analog of the Laplacian operator. To use
the Matern operator, one substitutes 

```math 
L = B^{(d)}(m, ϵ) = ((A^{(d)})^m - ϵ^2 I).
```

Importantly, $A$ is sparse, containing at most 5 nonzero entries
per row when $d = 2$ and $7$ nonzero entries per row when $d = 3$ and so on. The
Matern matrix $B^{(d)}(m, \epsilon)$ is also sparse, having $2(m+d)-1$ nonzero
entries per row. The sparsity of the matrix allows for the interpolation to
solve in linear time.


## Function Index

### One dimensional

```@docs
nablasq_1d_grid
matern_1d_grid
```

### Two dimensional

```@docs
bdy_nodes
nablasq_2d_grid
matern_2d_grid
```

### Three dimensional

```@docs
nablasq_3d_grid
matern_3d_grid
matern_w_punch
<!-- #Laplace_3D_Grid
#Parallel_Matern_3DGrid
#interp_nexus  -->
spdiagm_nonsquare
return_boundary_nodes
Matern3D_Grid
```

### Arbitrary dimensions

```@docs
nablasq_arb 
interp
```

### Spherical Punching

```@docs
punch_holes_3D
punch_holes_2D
punch_3D_cart
<!-- center_list  -->
```

