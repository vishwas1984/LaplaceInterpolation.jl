
"""

    nablasq_3d_grid(Nx,Ny, Nz, h, k, l, bc)

Construct the 3D Laplace matrix

# Arguments
  - `Nx::Int64`: The number of nodes in the first dimension
  - `Ny::Int64`: The number of nodes in the second dimension
  - `Nz::Int64`: The number of nodes in the third dimension
  - `h::Float64`: Grid spacing in the first dimension
  - `k::Float64`: Grid spacing in the second dimension
  - `l::Float64`: Grid spacing in the third dimension
  - `bc`: Boundary conditions (1 implies Neumann BC and 0 implies Do nothing BC)

# Outputs 

  - `-nablasq` (discrete Laplacian, real-symmetric positive-definite) on Nx×Ny×Nz grid

"""
function nablasq_3d_grid(Nx, Ny, Nz, h, k, l, bc)
    #haskey(A_matrix, (Nx, Ny, Nz, 1, 0.0, h, k, l, bc)) && return A_matrix[(Nx, Ny, Nz, 1, 0.0, h, k, l, bc)]

    o₁ = ones(Nx) / h
    del1 = spdiagm_nonsquare(Nx + 1, Nx, -1 => -o₁, 0 => o₁)
    o₂ = ones(Ny) / k
    del2 = spdiagm_nonsquare(Ny + 1, Ny, -1 => -o₂,0 => o₂)
    O3 = ones(Nz) / l
    del3 = spdiagm_nonsquare(Nz + 1, Nz, -1 => -O3, 0 => O3)
    Ax = del1'*del1
    Ay = del2'*del2
    Az = del3'*del3
    if (bc == 1)
      Ax[1,2] = -2.0/(h*h)
      Ax[Nx, Nx-1] = -2.0/(h*h)
      Ay[1,2] = -2.0/(k*k)
      Ay[Ny, Ny-1] = -2.0/(k*k)
      Az[1,2] = -2.0/(l*l)
      Az[Nz, Nz-1] = -2.0/(l*l)
      
      A3D = (kron(sparse(I, Nz, Nz), sparse(I, Ny, Ny), Ax) + 
            kron(sparse(I, Nz, Nz), Ay, sparse(I, Nx, Nx)) + 
            kron(Az, sparse(I, Ny, Ny), sparse(I, Nx, Nx)))
    else
      
      A3D = (kron(sparse(I, Nz, Nz), sparse(I, Ny, Ny), Ax) + 
            kron(sparse(I, Nz, Nz), Ay, sparse(I, Nx, Nx)) + 
            kron(Az, sparse(I, Ny, Ny), sparse(I, Nx, Nx)))

      BoundaryNodes, xneighbors, yneighbors, zneighbors =  return_boundary_nodes(Nx, Ny, Nz)
      count = 1
      for i in BoundaryNodes
          A3D[i, i] = 0.0
          A3D[i, i] = A3D[i, i] + xneighbors[count] / h ^ 2 + 
                       yneighbors[count] / k ^ 2 + zneighbors[count] / l ^ 2
          count = count + 1
      end
    end
    return A3D
end

""" Helper function to give the matern matrix """
function _Matern_matrix(Nx, Ny, Nz, m, eps, h, k, l, bc)
    #haskey(A_matrix, (Nx, Ny, Nz, m, eps, h, k, l, bc)) && return A_matrix[(Nx, Ny, Nz, m, eps, h, k, l, bc)]
    A3D = nablasq_3d_grid(Nx, Ny, Nz, h, k, l, bc) 
    sizeA = size(A3D, 1)
    for i = 1:sizeA
        A3D[i, i] = A3D[i, i] + eps^2
    end
    A3DMatern = A3D^m
    return A3DMatern
end

"""

    matern_3d_grid(imgg, discard, m, eps, h, k, l, bc)

Interpolates a single punch

# Arguments
  - `imgg`: the matrix containing the image
  - `discard::Union{Vector{CartesianIndex{3}}}, Vector{Int64}}`: the linear or 
       Cartesian indices of the values to be filled 
  - `m::Int64 = 1` : Matern parameter 
  - `eps::Float64 = 0.0`: Matern parameter eps
  - `h = 1.0`: Aspect ratio in the first dimension
  - `k = 1.0`: Aspect ratio in the second dimension
  - `l = 1.0`: Aspect ratio in the third dimension 
  - `bc`: Boundary conditions (1 implies Neumann BC and 0 implies Do nothing BC)

# Outputs
  - array containing the restored image

"""
function matern_3d_grid(imgg, discard::Union{Vector{CartesianIndex{3}}, Vector{Int64}},
                m::Int64 = 1,  eps::Float64 = 0.0, 
                h = 1.0, k = 1.0, l = 1.0, bc = 1) 
    Nx, Ny, Nz = size(imgg)
    A3D = (eps == 0.0)&&(m == 1) ? 
                nablasq_3d_grid(Nx, Ny, Nz, h, k, l, bc) :
                _Matern_matrix(Nx, Ny, Nz, m, eps, h, k, l, bc) 
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

    matern_w_punch(imgg, Qh_min, Qh_max, Qk_min, Qk_max, Ql_min, Ql_max, m, eps, h, k, l, symm)

Interpolate, in serial, multiple punches

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
interp = matern_w_punch(imgg, #Qh_min, Qh_max, Qk_min, Qk_max, Ql_min, Ql_max, radius,
                      radius, xpoints, ypoints, zpoints, m, eps, 
                      h, k, l, symm)
```
"""
function matern_w_punch(imgg, radius, xpoints, ypoints, zpoints,
                        m = 1, eps = 0.0, h = 1.0, k = 1.0, l = 1.0, symm = 'G';
                        return_punch_locs = false)
    Qh_min, Qh_max = get_Qw_maxmin(xpoints)
    Qk_min, Qk_max = get_Qw_maxmin(ypoints)
    Ql_min, Ql_max = get_Qw_maxmin(zpoints)
    radius_x, radius_y, radius_z = (typeof(radius) <: Tuple) ? radius : 
                                                (radius, radius, radius)
    dx, dy, dz = (xpoints[2] - xpoints[1], ypoints[2] - ypoints[1], 
                  zpoints[2] - zpoints[1])
    rpx, rpy, rpz = (Int64(round(radius_x/dx)) + m, Int64(round(radius_y/dy)) + m,
                  Int64(round(radius_z/dz))+ m)
    Nx, Ny, Nz = (length(xpoints), length(ypoints), length(zpoints))
    new_imgg = copy(imgg);
    exclusion_rule = center_check(symm)
    # find the first center
    (cpx, cpy, cpz) = (findmin(abs.(xpoints .- Qh_min))[2], 
                                  findmin(abs.(ypoints .- Qk_min))[2],
                                  findmin(abs.(zpoints .- Ql_min))[2])
    stridex, stridey, stridez = (-cpx, -cpy, -cpz) .+ (findmin(abs.(xpoints .- (Qh_min + 1)))[2], 
                                  findmin(abs.(ypoints .- (Qk_min + 1)))[2],
                                  findmin(abs.(zpoints .- (Ql_min + 1)))[2])
    punch_locs = return_punch_locs ? ones(Int64, size(imgg)) : nothing
    for (ih, h) in enumerate(Qh_min:Qh_max) 
        minpx = maximum([1, cpx + (ih - 1) * stridex - rpx])
        maxpx = minimum([cpx + (ih - 1) * stridex + rpx, Nx])
        for (ik, k) in enumerate(Qk_min:Qk_max) 
            minpy = maximum([1, cpy + (ik - 1) * stridey - rpy])
            maxpy = minimum([cpy + (ik - 1) * stridey + rpy, Ny])
            for (il, l) in enumerate(Ql_min:Ql_max)
                minpz = maximum([1, cpz + (il - 1) * stridez - rpz])
                maxpz = minimum([cpz + (il - 1) * stridez + rpz, Nz])
                if exclusion_rule(h, k, l)
                    d = punch_3D_cart((h,k,l), radius, xpoints[minpx:maxpx], 
                                      ypoints[minpy:maxpy], 
                                      zpoints[minpz:maxpz])
                    # Interpolate
                    new_imgg[minpx:maxpx, minpy:maxpy, minpz:maxpz] = matern_3d_grid(
                                imgg[minpx:maxpx, minpy:maxpy, minpz:maxpz], d, m, 
                                eps, h, k, l);
                    if return_punch_locs
                      punch_locs[d] .= 0
                    end
                end
            end
        end
    end
    if return_punch_locs
      return new_imgg, punch_locs
    else
      return new_imgg
    end
end

