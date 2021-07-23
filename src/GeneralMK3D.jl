
# functions: nablasq_3d_grid, return_boundary_nodes, 
# return_boundary_nodes_3D, punch_holes_nexus_Cartesian, 
# Matern_3d_Grid, Laplace_3D_grid,
# parallel_Matern_3DGrid, parallel_Laplace_3Dgrid

# Internal settings for caching
mutable struct Settings
  A_matrix_STORE_MAX::Int64
end

const SETTINGS = Settings(6)

# Dict for storing A_matrices
const A_matrix = Dict{Tuple{Int64, Int64, Int64, Int64, Float64, Float64, Float64,
                            Float64}, Matrix{Float64}}()
"""
  nablasq_3d_grid(Nx,Ny)

Construct the 3D Laplace matrix

# Arguments
  - `Nx::Int64`: The number of nodes in the first dimension
  - `Ny::Int64`: The number of nodes in the second dimension
  - `Nz::Int64`: The number of nodes in the third dimension
  - `h::Float64`: Grid spacing in the first dimension
  - `k::Float64`: Grid spacing in the second dimension
  - `l::Float64`: Grid spacing in the third dimension

# Outputs 

  - `-nablasq` (discrete Laplacian, real-symmetric positive-definite) on Nx×Ny grid

"""
function nablasq_3d_grid(Nx, Ny, Nz, h, k, l)
    haskey(A_matrix, (Nx, Ny, Nz, 1, 0.0, h, k, l)) && return A_matrix[(Nx, Ny, Nz, 1, 0.0, h, k, l)]
    o₁ = ones(Nx) / h
    del1 = spdiagm_nonsquare(Nx + 1, Nx, -1 => -o₁, 0 => o₁)
    o₂ = ones(Ny) / k
    del2 = spdiagm_nonsquare(Ny + 1, Ny, -1 => -o₂,0 => o₂)
    O3 = ones(Nz) / l
    del3 = spdiagm_nonsquare(Nz + 1, Nz, -1 => -O3, 0 => O3)
    A3D = (kron(sparse(I, Nz, Nz), sparse(I, Ny, Ny), del1'*del1) + 
            kron(sparse(I, Nz, Nz), del2' * del2, sparse(I, Nx, Nx)) + 
            kron(del3' * del3, sparse(I, Ny, Ny), sparse(I, Nx, Nx)))
    BoundaryNodes, xneighbors, yneighbors, zneighbors = 
            return_boundary_nodes(Nx, Ny, Nz)
    count = 1
    for i in BoundaryNodes
        A3D[i, i] = 0.0
        A3D[i, i] = A3D[i, i] + xneighbors[count] / h ^ 2 + 
                     yneighbors[count] / k ^ 2 + zneighbors[count] / l ^ 2
        count = count + 1
    end
    length(A_matrix) <  SETTINGS.A_matrix_STORE_MAX && (A_matrix[(Nx, Ny, Nz, 1, 0.0, h, k, l)] = A3D)
    #if length(A_matrix) == SETTINGS.A_matrix_STORE_MAX
    #  @warn "A_matrix cache full, no longer caching Laplace interpolation matrices."
    #end
    return A3D
end

""" Helper function to give the matern matrix """
function _Matern_matrix(Nx, Ny, Nz, m, eps, h, k, l)
    haskey(A_matrix, (Nx, Ny, Nz, m, eps, h, k, l)) && return A_matrix[(Nx, Ny, Nz, m, eps, h, k, l)]
    A3D = nablasq_3d_grid(Nx, Ny, Nz, h, k, l) 
    sizeA = size(A3D, 1)
    for i = 1:sizeA
        A3D[i, i] = A3D[i, i] + eps^2
    end
    A3DMatern = A3D^m
    length(A_matrix) <  SETTINGS.A_matrix_STORE_MAX && (A_matrix[(Nx, Ny, Nz, m, eps, h, k, l)] = A3DMatern)
    #if length(A_matrix) == SETTINGS.A_matrix_STORE_MAX
    #  @warn "A_matrix cache full, no longer caching Laplace interpolation matrices."
    #end
    A3DMatern
end

"""

  matern_3d_grid(imgg, discard, m, eps, h, k, l)

Interpolates a single punch

...
# Arguments
  - `imgg`: the matrix containing the image
  - `discard::Union{Vector{CartesianIndex{3}}}, Vector{Int64}}`: the linear or 
       Cartesian indices of the values to be filled 
  - `m::Int64 = 1` : Matern parameter 
  - `eps::Float64 = 0.0`: Matern parameter eps
  - `h = 1.0`: Aspect ratio in the first dimension
  - `k = 1.0`: Aspect ratio in the second dimension
  - `l = 1.0`: Aspect ratio in the third dimension 

# Outputs
  - array containing the restored image
...

"""
function matern_3d_grid(imgg, discard::Union{Vector{CartesianIndex{3}}, Vector{Int64}},
                m::Int64 = 1,  eps::Float64 = 0.0, 
                h = 1.0, k = 1.0, l = 1.0) 
    Nx, Ny, Nz = size(imgg)
    A3D = (eps == 0.0)&&(m == 1) ? 
                nablasq_3d_grid(Nx, Ny, Nz, h, k, l) :
                _Matern_matrix(Nx, Ny, Nz, m, eps, h, k, l) 
    totalsize = Nx * Ny * Nz
    C = sparse(I, totalsize, totalsize)
    rhs_a = copy(imgg)[:]
    for i in discard
        j = (typeof(i) <: CartesianIndex) ? LinearIndices(imgg)[i] : i
        C[j, j] = 0.0
        rhs_a[j] = 0.0
    end
    Id = sparse(I, totalsize, totalsize)    
    u = ((C - (Id - C) * A3D)) \ rhs_a
    return reshape(u, Nx, Ny, Nz)
end

# Add m pixels around the punch and then intersect with the size of the full
# image (3D only)
function pad_intersect(discard, m, Nx, Ny, Nz)
    B = bounding_box(discard)
    fi, li = (first(B) - CartesianIndex(m, m, m), last(B) + CartesianIndex(m, m, m))
    return intersect_box(fi:li, CartesianIndices((Nx,Ny,Nz)))
end

"""

  parallel_mat(imgg, Qh_min, Qh_max, Qk_min, Qk_max, Ql_min, Ql_max, m, eps, h, k, l, symm)

Interpolate, in parallel and in-place, multiple punches

...
# Arguments
  - `imgg`: the matrix containing the image
  - ` Qh_min, Qh_max, Qk_min, Qk_max, Ql_min, Ql_max::Int64`: the extents in h,k,l resp 
  - `m::Int64 = 1` : Matern parameter 
  - `eps::Float64 = 0.0`: Matern parameter eps
  - `h = 1.0`: Aspect ratio in the first dimension
  - `k = 1.0`: Aspect ratio in the second dimension
  - `l = 1.0`: Aspect ratio in the third dimension 

# Outputs
  - array containing the interpolated image

# Example

```<julia-repl>
h = k = l = 1.0
symm = 'A'
Nx = Ny = Nz = 61
radius = (0.5, 0.5, 0.5 )
Qh_min = Qk_min = Ql_min = -3.0
Qh_max = Qk_max = Ql_max = 3.0
xpoints = ypoints = zpoints = LinRange(Qh_min, Qh_max, Nx)
imgg = rand(Nx, Ny, Nz)
m = 1
eps = 0.0
interp = parallel_mat(imgg, Qh_min, Qh_max, Qk_min, Qk_max, Ql_min, Ql_max, radius,
                      xpoints, ypoints, zpoints, m, eps, 
                      h, k, l, symm)
```

...
"""
function parallel_mat(imgg, Qh_min, Qh_max, Qk_min, Qk_max, Ql_min, Ql_max, radius,
                      xpoints, ypoints, zpoints,
                        m = 1, eps = 0.0, h = 1.0, k = 1.0, l = 1.0, symm = 'G')
    centers = center_list(symm, Qh_min, Qh_max, Qk_min, Qk_max, Ql_min, Ql_max)
    new_imgg = copy(imgg)
    # Threads.@threads for c in centers
    for c in centers
        discard = LaplaceInterpolation.punch_3D_cart(c, radius, xpoints, ypoints, zpoints)
        selection = LaplaceInterpolation.pad_intersect(discard, m, size(imgg)...)
        fi = first(selection)
        d_shift = map(d -> d - fi + CartesianIndex(1,1,1), discard)
        # Interpolate
        new_imgg[selection] = matern_3d_grid(imgg[selection], d_shift, m, eps, h, k, l)
    end
    return new_imgg
end

