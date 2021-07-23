
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
    if length(A_matrix) == SETTINGS.A_matrix_STORE_MAX && SETTINGS.VERBOSE
      @warn "A_matrix cache full, no longer caching Laplace interpolation matrices."
    end
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
    if length(A_matrix) == SETTINGS.A_matrix_STORE_MAX && SETTINGS.VERBOSE
      @warn "A_matrix cache full, no longer caching Laplace interpolation matrices."
    end
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
...
"""
function parallel_mat(imgg, Qh_min, Qh_max, Qk_min, Qk_max, Ql_min, Ql_max,
                        m = 1, eps = 0.0, h = 1.0, k = 1.0, l = 1.0, symm = 'G')
    centers = center_list(symm, Qh_min, Qh_max, Qk_min, Qk_max, Ql_min, Ql_max)
    discard = punch_3Dd_cart.(centers, radius, x, y, z)
    # Threads.@threads for d in discard
    for d in discard
        fi, li = (first(d), last(d) + CartesianIndex(m, m, m))
        selection = map(i -> i - fi + CartesianIndex(m, m, m), d)
        # Interpolate
        imgg[fi:li] = interp(xpoints[fi[1]:li[1]], ypoints[fi[2]:li[2]], 
                zpoints[fi[3]:li[3]], 
                imgg[fi:li], 
                selection, eps, m)
    end
    return imgg
end

