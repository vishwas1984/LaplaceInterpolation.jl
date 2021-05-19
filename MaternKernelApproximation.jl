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
        A3D[i, i] = A3D[i, i] + xneighbors[count] / h ^ 2 + yneighbors[count] / k ^ 2 + zneighbors[count] / l ^ 2
        count = count + 1
    end
    return A3D
end

"""
  return_boundary_nodes(xpoints, ypoints, zpoints)

...
# Arguments

  - `xpoints::Vector{T} where T<:Real`: the vector containing the x coordinate
  - `ypoints::Vector{T} where T<:Real`: the vector containing the y coordinate
  - `zpoints::Vector{T} where T<:Real`: the vector containing the z coordinate
...

...
# Outputs
  - `BoundaryNodes3D::Vector{Int64}`: vector containing the indices of coordinates 
  on the boundary of the rectangular 3D volume
...

"""
function return_boundary_nodes(xpoints, ypoints, zpoints)
    BoundaryNodes3D =[]
    xneighbors = []
    yneighbors = []
    zneighbors = []
    counter = 0
    for k = 1:zpoints
        for j = 1:ypoints
            for i = 1:xpoints
                counter=counter+1
                if(k == 1 || k == zpoints || j == 1|| j == ypoints || i == 1 || i == xpoints)
                    BoundaryNodes3D = push!(BoundaryNodes3D, counter)
                    if(k == 1 || k == zpoints)
                        push!(zneighbors, 1)
                    else
                        push!(zneighbors, 2)
                    end
                    if(j == 1 || j == ypoints)
                        push!(yneighbors, 1)
                    else
                        push!(yneighbors, 2)
                    end
                    if(i == 1 || i == xpoints)
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
  punch_holes_nexus(xpoints, ypoints, zpoints, radius)

...
# Arguments

  - `xpoints::Vector{T} where T<:Real`: the vector containing the x coordinate
  - `ypoints::Vector{T} where T<:Real`: the vector containing the y coordinate
  - `zpoints::Vector{T} where T<:Real`: the vector containing the z coordinate
  - `radius::Union{Float64,Vector{Float64}}`: the radius, or radii of the punch, if vector.
...

...
# Outputs


  - `absolute_indices::Vector{Int64}`: vector containing the indices of coordinates 
  inside the punch

...
"""
function punch_holes_nexus(xpoints, ypoints, zpoints, radius)
    rad = (typeof(radius) <: Tuple) ? radius : (radius, radius, radius)
    radius_x, radius_y, radius_z = rad 
    absolute_indices = Int64[]
    count = 1
    for i = 1:length(zpoints)
        ir = round(zpoints[i])
        for j = 1:length(ypoints)
            jr = round(ypoints[j])
            for h = 1:length(xpoints)
                hr = round(xpoints[h])
                if (((hr - xpoints[h])/radius_x)^2 + ((jr - ypoints[j])/radius_y^2) 
                    + ((ir - zpoints[i])/radius_z)^2 <= 1.0)
                    append!(absolute_indices, count)
                    count += 1
                end
            end
        end
    end
    return absolute_indices
end

"""

  Matern3D_Grid(xpoints, ypoints, zpoints, imgg, epsilon, radius, h, k, l, m)

...
# Arguments
  - `xpoints::Vector{T} where T<:Real`: the vector containing the x coordinate
  - `ypoints::Vector{T} where T<:Real`: the vector containing the y coordinate
  - `zpoints::Vector{T} where T<:Real`: the vector containing the z coordinate
  - `imgg`: the matrix containing the image
  - `epsilon`: Matern parameter epsilon
  - `radius::Vector`: the tuple containing the punch radii 
  - `h::Float`: grid spacing along the x-axis
  - `k::Float`: grid spacing along the y-axis
  - `l::Float`: grid spacing along the z-axis
  - `m::Int` : Matern parameter 

# Outputs
  - tuple containing the restored image and the punched image.
...

"""
function Matern3D_Grid(xpoints, ypoints, zpoints, imgg, epsilon, radius, h, k, l, m)
    A3D = ∇²3d_Grid(length(xpoints), length(ypoints), length(zpoints), h, k, l)
    sizeA = size(A3D, 1)
    for i = 1:sizeA
        A3D[i, i] = A3D[i, i] + epsilon^2
    end
    A3DMatern = A3D
    for i = 1:m - 1
        A3DMatern = A3DMatern * A3D
    end
    discard = punch_holes_nexus(xpoints, ypoints, zpoints, radius)
    punched_image = copy(imgg)
    punched_image[discard] .= 1
    totalsize = prod(size(imgg))
    C = sparse(I, totalsize, totalsize)
    rhs_a = punched_image[:]
    for i in discard
        C[i, i] = 0
        rhs_a[i] = 0
    end
    Id = sparse(I, totalsize, totalsize)    
    u = ((C - (Id - C) * A3DMatern)) \ rhs_a
    return (u, punched_image[:])
end

"""

  Laplace3D_Grid(xpoints, ypoints, zpoints, imgg, radius, h, k, l)

...
# Arguments
  - `xpoints::Vector{T} where T<:Real`: the vector containing the x coordinate
  - `ypoints::Vector{T} where T<:Real`: the vector containing the y coordinate
  - `zpoints::Vector{T} where T<:Real`: the vector containing the z coordinate
  - `imgg`: the matrix containing the image
  - `radius::Vector`: the tuple containing the punch radii 
  - `h::Float`: grid spacing along the x-axis
  - `k::Float`: grid spacing along the y-axis
  - `l::Float`: grid spacing along the z-axis

# Outputs
  - tuple containing the restored image and the punched image.
...

"""
function Laplace3D_Grid(xpoints, ypoints, zpoints, imgg, radius, h, k, l)
    A3D = ∇²3d_Grid(length(xpoints), length(ypoints), length(zpoints), h, k, l)
    discard = punch_holes_nexus(xpoints, ypoints, zpoints, radius)
    punched_image = copy(imgg)
    punched_image[discard] .= 1
    totalsize = prod(size(imgg))
    C = sparse(I, totalsize, totalsize)
    rhs_a = punched_image[:]
    for i in discard
        C[i,i] = 0
        rhs_a[i] = 0
    end
    Id = sparse(I, totalsize, totalsize)
    u = ((C - (Id - C) * A3D)) \ rhs_a
    return u, punched_image[:]
end

"""

  Parallel_Laplace3D_Grid(xpoints, ypoints, zpoints, imgg, radius, h, k, l,
          xmin, xmax, ymin, ymax, zmin, zmax)

Compute the spherically-punched, Laplace-interpolated 3D data

...
# Arguments
  - `xpoints::Vector{T} where T<:Real`: the vector containing the x coordinate
  - `ypoints::Vector{T} where T<:Real`: the vector containing the y coordinate
  - `zpoints::Vector{T} where T<:Real`: the vector containing the z coordinate
  - `imgg`: the matrix containing the image
  - `radius::Vector`: the tuple containing the punch radii 
  - `h::Float`: grid spacing along the x-axis
  - `k::Float`: grid spacing along the y-axis
  - `l::Float`: grid spacing along the z-axis
  - `xmin::Int64`: Vishwas should fill in the next six fields. 
  - `xmax::Int64`:
  - `ymin::Int64`:
  - `ymax::Int64`:
  - `zmin::Int64`:
  - `zmax::Int64`:

# Outputs
  - array containing the restored image 

# Example 

```julia-repl
(xmin, xmax) = (ymin, ymax) = (zmin,zmax) = (0.0, 1.0)
xpoints = ypoints = zpoints = -0.2:0.2:1.2
h = k = l = 0.2
imgg = randn(8,8,8)
radius = 0.2
restored = Parallel_Laplace3D_Grid(xpoints, ypoints, zpoints, imgg, radius, 
                                 h, k, l, xmin, xmax, ymin, ymax, zmin, zmax)
```

...

"""
function Parallel_Laplace3D_Grid(xpoints::Union{StepRangeLen{T},Vector{T}}, 
                                ypoints::Union{StepRangeLen{T},Vector{T}}, 
                                zpoints::Union{StepRangeLen{T},Vector{T}}, 
                                imgg::Array{P,3}, 
                                radius::Union{Q,Tuple{Q,Q,Q}}, 
                                h::Float64, k::Float64, l::Float64, 
                                xmin::R, xmax::R, ymin::R, ymax::R, zmin::R, zmax::R 
                                ) where{T<:Number,P<:Number,Q<:Number,R<:Number}
  # 
  fun(x,y,z,w) = Int(round((x -y)/z) - w ) 
  ran(x,y,z,w) = fun(x, y, z, w) .+ (0, 2*w + 1)
  #
  rad = (typeof(radius) <: Tuple) ? radius : (radius, radius, radius)
  radius_x, radius_y, radius_z = rad 
  stride_h = Int64(round(radius_x / h))
  stride_k = Int64(round(radius_y / k))
  stride_l = Int64(round(radius_z / l))
  cartesian_product_boxes = [(ran(i, xpoints[1], h, stride_h)..., 
                              ran(j, ypoints[1], k, stride_k)...,
                              ran(kk, zpoints[1], l, stride_l)...)  
                         for i in xmin:xmax for j in ymin:ymax for kk in zmin:zmax]
  #
  z3d_restored = copy(imgg)
  #
  Threads.@threads for i = 1:length(cartesian_product_boxes)
    i1, i2, j1, j2, k1, k2 = cartesian_product_boxes[i]
    restored_img = Laplace3D_Grid(xpoints[i1 + 1:i2], ypoints[j1 + 1:j2], 
                           zpoints[k1 + 1:k2], 
                           imgg[i1 + 1:i2, j1 + 1:j2, k1 + 1:k2], 
                           radius, h, k, l)[1]
    z3d_restored[i1 + 1:i2, j1 + 1:j2, k1 + 1:k2] = reshape(restored_img, 
                           (2 * stride_h + 1, 2 * stride_k + 1, 2 * stride_l + 1))
  end
  #
  return z3d_restored[:]
  #
end

"""

  Parallel_Matern3D_Grid(xpoints, ypoints, zpoints, imgg, epsilon, radius, h, k, l,
          xmin, xmax, ymin, ymax, zmin, zmax, m)

Compute the spherically-punched, Matern-interpolated 3D data

...
# Arguments
  - `xpoints::Vector{T} where T<:Real`: the vector containing the x coordinate
  - `ypoints::Vector{T} where T<:Real`: the vector containing the y coordinate
  - `zpoints::Vector{T} where T<:Real`: the vector containing the z coordinate
  - `imgg`: the matrix containing the image
  - `epsilon`: one of the matern parameters
  - `radius::Vector`: the tuple containing the punch radii 
  - `h::Float`: grid spacing along the x-axis
  - `k::Float`: grid spacing along the y-axis
  - `l::Float`: grid spacing along the z-axis
  - `xmin::Int64`: Vishwas should fill in the next six fields. 
  - `xmax::Int64`:
  - `ymin::Int64`:
  - `ymax::Int64`:
  - `zmin::Int64`:
  - `zmax::Int64`:
  - `m::Int64`: The matern order parameter 

# Outputs
  - array containing the restored image 

# Example 

```julia-repl
(xmin, xmax) = (ymin, ymax) = (zmin,zmax) = (0.0, 1.0)
xpoints = ypoints = zpoints = -0.2:0.2:1.2
h = k = l = 0.2
imgg = randn(8,8,8)
m = 2
epsilon = 0.0
radius = 0.2
restored = Parallel_Matern3D_Grid(xpoints, ypoints, zpoints, imgg, epsilon, radius, 
                                  h, k, l, xmin, xmax, ymin, ymax, zmin, zmax, m)
```

...

"""
function Parallel_Matern3D_Grid(xpoints::Union{StepRangeLen{T},Vector{T}}, 
                                ypoints::Union{StepRangeLen{T},Vector{T}}, 
                                zpoints::Union{StepRangeLen{T},Vector{T}}, 
                                imgg::Array{P,3}, epsilon::Q, 
                                radius::Union{Q,Tuple{Q,Q,Q}}, 
                                h::Float64, k::Float64, l::Float64, 
                                xmin::R, xmax::R, ymin::R, ymax::R, zmin::R, zmax::R, 
                                m::Int) where{T<:Number,P<:Number,Q<:Number,R<:Number}
  # 
  fun(x,y,z,w) = Int(round((x -y)/z) - w ) 
  ran(x,y,z,w) = fun(x, y, z, w) .+ (0, 2*w + 1)
  #
  rad = (typeof(radius) <: Tuple) ? radius : (radius, radius, radius)
  radius_x, radius_y, radius_z = rad 
  stride_h = Int64(round(radius_x / h))
  stride_k = Int64(round(radius_y / k))
  stride_l = Int64(round(radius_z / l))
  cartesian_product_boxes = [(ran(i, xpoints[1], h, stride_h)..., 
                              ran(j, ypoints[1], k, stride_k)...,
                              ran(kk, zpoints[1], l, stride_l)...)  
                         for i in xmin:xmax for j in ymin:ymax for kk in zmin:zmax]
  #
  z3d_restored = copy(imgg)
  #
  Threads.@threads for i = 1:length(cartesian_product_boxes)
    i1, i2, j1, j2, k1, k2 = cartesian_product_boxes[i]
    restored_img = Matern3D_Grid(xpoints[i1 + 1:i2], ypoints[j1 + 1:j2], 
                           zpoints[k1 + 1:k2], 
                           imgg[i1 + 1:i2, j1 + 1:j2, k1 + 1:k2], 
                           epsilon, radius, h, k, l, m)[1]
    z3d_restored[i1 + 1:i2, j1 + 1:j2, k1 + 1:k2] = reshape(restored_img, 
                           (2 * stride_h + 1, 2 * stride_k + 1, 2 * stride_l + 1))
  end
  #
  return z3d_restored[:]
  #
end

