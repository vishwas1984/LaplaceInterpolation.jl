
# functions: nablasq_3d_grid, return_boundary_nodes, 
# return_boundary_nodes_3D, punch_holes_nexus_Cartesian, 
# Matern_3d_Grid, Laplace_3D_grid,
# parallel_Matern_3DGrid, parallel_Laplace_3Dgrid

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
    return A3D
end

""" Helper function to give the matern matrix """
function _Matern_matrix(Nx, Ny, Nz, m, epsilon, h, k, l)
    A3D = nablasq_3d_grid(Nx, Ny, Nz, h, k, l) 
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

  matern_3d_grid(imgg, discard, m, epsilon, h, k, l)

Interpolates a single punch

...
# Arguments
  - `imgg`: the matrix containing the image
  - `discard::Vector{Int64}`: the linear indices of the values to be filled 
  - `m::Int64 = 1` : Matern parameter 
  - `epsilon::Float64 = 0.0`: Matern parameter epsilon
  - `h = 1.0`: Aspect ratio in the first dimension
  - `k = 1.0`: Aspect ratio in the second dimension
  - `l = 1.0`: Aspect ratio in the third dimension 

# Outputs
  - array containing the restored image
...

"""
function matern_3d_grid(imgg, discard::Union{Vector{CartesianIndex{3}}, Vector{Int64}},
                m::Int64 = 1,  epsilon::Float64 = 0.0, 
                h = 1.0, k = 1.0, l = 1.0) 
    Nx, Ny, Nz = size(imgg)
    A3D = (epsilon == 0.0)&&(m == 1) ? 
                nablasq_3d_grid(Nx, Ny, Nz, h, k, l) :
                _Matern_matrix(Nx, Ny, Nz, m, epsilon, h, k, l) 
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


