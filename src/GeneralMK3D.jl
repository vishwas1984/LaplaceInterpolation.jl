
# functions: ∇²3d_Grid, return_boundary_nodes, 
# return_boundary_nodes_3D, punch_holes_nexus_Cartesian, 
# Matern_3d_Grid, Laplace_3D_grid,
# parallel_Matern_3DGrid, parallel_Laplace_3Dgrid

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
        A3D[i, i] = A3D[i, i] + xneighbors[count] / h ^ 2 + 
                     yneighbors[count] / k ^ 2 + zneighbors[count] / l ^ 2
        count = count + 1
    end
    return A3D
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
  punch_holes_nexus_Cartesian(x, y, z, radius)

...
# Arguments

  - `x::Vector{T} where T<:Real`: the vector containing the x coordinate
  - `y::Vector{T} where T<:Real`: the vector containing the y coordinate
  - `z::Vector{T} where T<:Real`: the vector containing the z coordinate
  - `radius::Union{Float64,Tuple{Float64}}`: the radius, or radii of the punch, if vector.
...

...
# Outputs

  - `inds::Vector{Int64}`: vector containing the indices of coordinates 
  inside the punch

...
"""
function punch_holes_nexus_Cartesian(x, y, z, radius)
    radius_x, radius_y, radius_z = (typeof(radius) <: Tuple) ? radius : (radius, radius, radius)
    inds = filter(i -> (((x[i[1]] - round(x[i[1]])) / radius_x) ^2 
                        + ((y[i[2]] - round(y[i[2]])) / radius_y) ^2 
                        + ((z[i[3]] - round(z[i[3]])) / radius_z) ^2 <= 1.0),
                  CartesianIndices((1:length(x), 1:length(y), 1:length(z))))
    return inds
end

function _Matern_matrix(Nx, Ny, Nz, dx, dy, dz, epsilon, m)
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

Interpolates a single punch

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
  - array containing the restored image
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
                               y[2] - y[1], z[2] - z[1], epsilon, m) 
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


"""

  parallel_interp!(x, y, z, imgg, discard, epsilon, m)

Interpolate, in parallel and in-place, multiple punches

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
  - array containing the interpolated image 
...
"""
function parallel_interp!(x, y, z, imgg,
                        discard::Vector{Vector{CartesianIndex{3}}},
                        epsilon = 0.0, m = 1)
    Threads.@threads for d in discard
        fi, li = (first(d), last(d) + CartesianIndex(1, 1, 1))
        selection = map(i -> i - fi + CartesianIndex(1, 1, 1), d)
        # Interpolate
        imgg[fi:li] = interp(xpoints[fi[1]:li[1]], ypoints[fi[2]:li[2]], 
                zpoints[fi[3]:li[3]], 
                imgg[fi:li], 
                selection, epsilon, m)
    end
end

"""

  interp_1lu(x, y, z, imgg, punch_template, epsilon, m)

Interpolate around the bragg peaks in the image by tiling the punch_template

...
# Arguments
  - `x::Vector{T} where T<:Real`: the vector containing the x coordinate
  - `y::Vector{T} where T<:Real`: the vector containing the y coordinate
  - `z::Vector{T} where T<:Real`: the vector containing the z coordinate
  - `imgg::Array{Float64,3]`: the matrix containing the image
  - `punch_template::Array{Float64,3}`: one lattice unit template of the values to be filled 
  - `epsilon::Float64 = 0.0`: Matern parameter epsilon
  - `m::Int64 = 1` : Matern parameter 

# Outputs
  - array containing the interpolated image 
...
"""
function interp_1lu(x, y, z, imgg, punch_template, epsilon, m)
  discard = findall(punch_template .> 0)
  res = interp(x, y, z, imgg, discard, epsilon, m)
  return res
end

"""

  interp_nexus(x, y, z, imgg, punch_template, epsilon, m)

Interpolate around the bragg peaks in the image by tiling the punch_template

...
# Arguments
  - `x::Vector{T} where T<:Real`: the vector containing the x coordinate
  - `y::Vector{T} where T<:Real`: the vector containing the y coordinate
  - `z::Vector{T} where T<:Real`: the vector containing the z coordinate
  - `imgg::Array{Float64,3]`: the matrix containing the image
  - `punch_template::Array{Float64,3}`: one lattice unit template of the values to be filled 
  - `epsilon::Float64 = 0.0`: Matern parameter epsilon
  - `m::Int64 = 1` : Matern parameter 

# Outputs
  - array containing the interpolated image 
...

# Example

```<julia-repl>
x = y = z = collect(-0.5:0.1:10)
imgg = randn(length(x), length(y), length(z))
punch_template = zeros(10, 10, 10); punch_template[5, 5, 5] = 1
interp_nexus(x, y, z, imgg, punch_template)
```
"""
function interp_nexus(x, y, z, imgg, punch_template, epsilon = 0.0, m = 1)
  discard = findall(punch_template .> 0)
  x_unit, y_unit, z_unit = size(punch_template)
  Nx, Ny, Nz = size(imgg)
  Qx = ceil(x[end]) + 1 
  Qy = ceil(y[end]) + 1
  Qz = ceil(z[end]) + 1
  corners_x = findall((x .- 0.5) .% 1 .== 0.0)
  corners_y = findall((y .- 0.5) .% 1 .== 0.0)
  corners_z = findall((z .- 0.5) .% 1 .== 0.0)
  corners = [CartesianIndex(xind, yind, zind) for xind in corners_x[1:end-1] 
                                              for yind in corners_y[1:end-1]
                                              for zind in corners_y[1:end-1]]
  # punched = copy(imgg);
  # Threads.@threads for (i,c) in enumerate(corners)
  for (i,c) in enumerate(corners)
    tcx = minimum([Nx, Tuple(c)[1] + x_unit])
    tcy = minimum([Ny, Tuple(c)[2] + y_unit])
    tcz = minimum([Nz, Tuple(c)[3] + z_unit])
    tc = CartesianIndex(tcx, tcy, tcz)
    imgg[c:tc] = interp(x[c[1]:tc[1]], y[c[2]:tc[2]], z[c[3]:tc[3]], 
                       imgg[c:tc], discard, Float64(epsilon), Int64(m))
    # punched[c:tc] .= 0.0
  end
  return imgg
end


