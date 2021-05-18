# This code interpolates for the missing points in an image. The code is
# specifically designed for removing Bragg peaks using the punch and fill
# algorithm. This code needs the image and the coordinates where the Bragg
# peaks needs to be removed and the radius (which can be the approximate width
# of the peaks). The code assumes all the "punches" will be of the same size
# and there are no Bragg peaks on the boundaries. Lines 2 to ~ 175 consists of
# helper functions and 175 onwards corresponds to the driver code.

# functions: spdiagm_nonsquare, ∇²3d_Grid, return_boundary_nodes, 
# return_boundary_nodes_3D, punch_holes_nexus, Matern_3d_Grid, Laplace_3D_grid,
# parallel_Matern_3DGrid, parallel_Laplace_3Dgrid

using LinearAlgebra, SparseArrays

"""
  spdiagm_nonsquare(m, n, args...)

Construct a sparse diagonal matrix from Pairs of vectors and diagonals. Each
vector arg.second will be placed on the arg.first diagonal. By default (if
size=nothing), the matrix is square and its size is inferred from kv, but a
non-square size m×n (padded with zeros as needed) can be specified by passing
m,n as the first arguments.

# Arguments
  - `m::Int64`: First dimension of the output matrix
  - `n::Int64`: Second dimension of the output matrix
  - `args::Tuple{T} where T<:Pair{<:Integer,<:AbstractVector}` 

# Outputs 

  - sparse matrix of size mxn containing the values in args 

"""
function spdiagm_nonsquare(m, n, args...)
    I, J, V = SparseArrays.spdiagm_internal(args...)
    return sparse(I, J, V, m, n)
end

"""
  ∇²3d_Grid(n₁,n₂)

Construct the 3D Laplace matrix

# Arguments
  - `n₁::Int64`: The number of nodes in the first dimension
  - `n₂::Int64`: The number of nodes in the second dimension
  - `n3::Int64`: The number of nodes in the third dimension
  - `h::Float64`: Grid spacing in the first dimension
  - `k::Float64`: Grid spacing in the second dimension
  - `l::Float64`: Grid spacing in the third dimension

# Outputs 

  - `-∇²` (discrete Laplacian, real-symmetric positive-definite) on n₁×n₂ grid

"""
function ∇²3d_Grid(n₁, n₂, n3, h, k, l)
    o₁ = ones(n₁) / h
    ∂₁ = spdiagm_nonsquare(n₁ + 1, n₁, -1 => -o₁, 0 => o₁)
    o₂ = ones(n₂) / k
    ∂₂ = spdiagm_nonsquare(n₂ + 1, n₂, -1 => -o₂,0 => o₂)
    O3 = ones(n3) / l
    del3 = spdiagm_nonsquare(n3 + 1, n3, -1 => -O3, 0 => O3)
    A3D = (kron(sparse(I, n3, n3), sparse(I, n₂, n₂), ∂₁'*∂₁) + 
            kron(sparse(I, n3, n3), ∂₂' * ∂₂, sparse(I, n₁, n₁)) + 
            kron(del3' * del3, sparse(I, n₂, n₂), sparse(I, n₁, n₁)))
    BoundaryNodes, xneighbors, yneighbors, zneighbors = 
            return_boundary_nodes(n₁, n₂, n3)
    count = 1
    for i in BoundaryNodes
        A3D[i, i] = 0.0
        A3D[i, i] = A3D[i, i] + xneighbors[count] / h ^ 2 
                    + yneighbors[count] / k ^ 2 + zneighbors[count] / l ^ 2
        count = count + 1
    end
    return A3D
end

"""
  return_boundary_nodes(x, y, z)

...
# Arguments

  - `x::Vector{T} where T<:Real`: the vector containing the x coordinate
  - `y::Vector{T} where T<:Real`: the vector containing the y coordinate
  - `z::Vector{T} where T<:Real`: the vector containing the z coordinate
...

...
# Outputs
  - `BoundaryNodes3D::Vector{Int64}`: vector containing the indices of coordinates 
  on the boundary of the rectangular 3D volume
...

"""
function return_boundary_nodes(x, y, z)
    BoundaryNodes3D =[]
    xneighbors = []
    yneighbors = []
    zneighbors = []
    counter = 0
    for k = 1:z
        for j = 1:y
            for i = 1:x
                counter=counter+1
                if(k == 1 || k == z || j == 1|| j == y || i == 1 || i == x)
                    BoundaryNodes3D = push!(BoundaryNodes3D, counter)
                    if(k == 1 || k == z)
                        push!(zneighbors, 1)
                    else
                        push!(zneighbors, 2)
                    end
                    if(j == 1 || j == y)
                        push!(yneighbors, 1)
                    else
                        push!(yneighbors, 2)
                    end
                    if(i == 1 || i == x)
                        push!(xneighbors, 1)
                    else
                        push!(xneighbors, 2)
                    end
                end
            end
        end
    end
    return BoundaryNodes3D, xneighbors, yneighbors, zneighbors
end

"""
  punch_hole_3D(center, radius, xpoints, ypoints, zpoints)

...
# Arguments

  - `center::Tuple{T}`: the tuple containing the center of a round punch
  - `radius::Union{Tuple{Float64},Float64}`: the radii/radius of the punch
  - `x::Vector{T} where T<:Real`: the vector containing the x coordinate
  - `y::Vector{T} where T<:Real`: the vector containing the y coordinate
  - `z::Vector{T} where T<:Real`: the vector containing the z coordinate
...

...
# Outputs
  - `inds::Vector{Int64}`: vector containing the indices of coordinates 
  inside the punch
  - `bbox::Tuple{Int64}`: the bounding box coordinates of the smallest box to fit around the punch
...

"""
function punch_hole_3D(center, radius, x, y, z)
    radius_x, radius_y, radius_z = (typeof(radius) <: Tuple) ? radius : 
                                                (radius, radius, radius)
    inds = filter(i -> (((x[i[1]]-center[1])/radius_x)^2 
                        + ((y[i[2]]-center[2])/radius_y)^2 
                        + ((z[i[3]] - center[3])/radius_z)^2 <= 1.0),
                  CartesianIndices((1:length(x), 1:length(y), 1:length(z))))
    (length(inds) == 0) && error("Empty punch.")
    return inds
end

"""
  punch_holes_nexus(x, y, z, radius)

...
# Arguments

  - `x::Vector{T} where T<:Real`: the vector containing the x coordinate
  - `y::Vector{T} where T<:Real`: the vector containing the y coordinate
  - `z::Vector{T} where T<:Real`: the vector containing the z coordinate
  - `radius::Union{Float64,Vector{Float64}}`: the radius, or radii of the punch, if vector.
...

...
# Outputs


  - `inds::Vector{Int64}`: vector containing the indices of coordinates 
  inside the punch

...
"""
function punch_holes_nexus(x, y, z, radius)
    radius_x, radius_y, radius_z = (typeof(radius) <: Tuple) ? radius : (radius, radius, radius)
    inds = filter(i -> (((x[i[1]] - round(x[i[1]])) / radius_x) ^2 
                        + ((y[i[2]] - round(y[i[2]])) / radius_y) ^2 
                        + ((z[i[3]] - round(z[i[3]])) / radius_z) ^2 <= 1.0),
                  CartesianIndices((1:length(x), 1:length(y), 1:length(z))))
    return inds
end

function _Matern_matrix(Nx, Ny, Nz, dx, dy, dz)
    A3D = ∇²3d_Grid(Nx, Ny, Nz, dx, dy, dz) 
    sizeA = size(A3D, 1)
    for i = 1:sizeA
        A3D[i, i] = A3D[i, i] + epsilon^2
    end
    A3DMatern = A3D
    for i = 1:m - 1
        A3DMatern = A3DMatern * A3D
    end
    return A3DMatern
end

"""

  interp(x, y, z, imgg, discard, epsilon, m)

...
# Arguments
  - `x::Vector{T} where T<:Real`: the vector containing the x coordinate
  - `y::Vector{T} where T<:Real`: the vector containing the y coordinate
  - `z::Vector{T} where T<:Real`: the vector containing the z coordinate
  - `imgg`: the matrix containing the image
  - `discard::Vector{Int64}`: the linear indices of the values to be filled 
  - `epsilon::Float64 = 0.0`: Matern parameter epsilon
  - `m::Int64 = 1` : Matern parameter 

# Outputs
  - tuple containing the restored image and the punched image.
...

"""
function interp(x, y, z, imgg, discard::Vector{CartesianIndex{3}},
                epsilon::Float64 = 0.0, m::Int64 = 1)
    ((length(x) !== size(imgg, 1)) || (length(y) !== size(imgg, 2)) || 
        (length(z) !== size(imgg,3))) && error("Axes lengths must match image dims")
    A3D = (epsilon == 0.0)&&(m == 1) ? 
                ∇²3d_Grid(length(x), length(y), length(z), x[2] - x[1],
                               y[2] - y[1], z[2] - z[1]) :
                _Matern_matrix(length(x), length(y), length(z), x[2] - x[1], 
                               y[2] - y[1], z[2] - z[1]) 
    totalsize = prod(size(imgg))
    C = sparse(I, totalsize, totalsize)
    rhs_a = copy(imgg)[:]
    for i in discard
        j = LinearIndices(imgg)[i]
        C[j, j] = 0.0
        rhs_a[j] = 0.0
    end
    Id = sparse(I, totalsize, totalsize)    
    u = ((C - (Id - C) * A3D)) \ rhs_a
    return u
end

