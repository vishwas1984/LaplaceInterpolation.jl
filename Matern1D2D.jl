# This code interpolates for the missing points in an image. The code is specifically
# designed for removing Bragg peaks using the punch and fill algorithm. This code
# needs the image and the coordinates where the Bragg peaks needs to be removed and
# the radius (which can be the approximate width of the peaks). The code assumes all
# the "punches" will be of the same size and there are no Bragg peaks on the
# boundaries. Lines 2 to ~ 175 consists of helper functions and 175 onwards
# corresponds to the driver code.

using LinearAlgebra, SparseArrays


"""
  ∇²(n₁,n₂)

Construct the 2D Laplace matrix

# Arguments
  - `n₁::Int64`: The number of nodes in the first dimension
  - `n₂::Int64`: The number of nodes in the second dimension

# Outputs 

  - `-∇²` (discrete Laplacian, real-symmetric positive-definite) on n₁×n₂ grid

"""
function ∇²(n₁,n₂)
    o₁ = ones(n₁)
    ∂₁ = spdiagm_nonsquare(n₁+1,n₁,-1=>-o₁,0=>o₁)
    o₂ = ones(n₂)
    ∂₂ = spdiagm_nonsquare(n₂+1,n₂,-1=>-o₂,0=>o₂)
    return kron(sparse(I,n₂,n₂), ∂₁'*∂₁) + kron(∂₂'*∂₂, sparse(I,n₁,n₁))
end

"""
  ∇²3d(n₁,n₂)

Construct the 3D Laplace matrix

# Arguments
  - `n₁::Int64`: The number of nodes in the first dimension
  - `n₂::Int64`: The number of nodes in the second dimension
  - `n3::Int64`: The number of nodes in the third dimension

# Outputs 

  - `-∇²` (discrete Laplacian, real-symmetric positive-definite) on n₁×n₂ grid

"""
function ∇²3d(n₁,n₂,n3)
    o₁ = ones(n₁)
    ∂₁ = spdiagm_nonsquare(n₁+1,n₁,-1=>-o₁,0=>o₁)
    o₂ = ones(n₂)
    ∂₂ = spdiagm_nonsquare(n₂+1,n₂,-1=>-o₂,0=>o₂)
    O3 = ones(n3)
    del3 = spdiagm_nonsquare(n3+1,n3,-1=>-O3,0=>O3)
    return kron(sparse(I,n3,n3),sparse(I,n₂,n₂), ∂₁'*∂₁) + 
           kron(sparse(I,n3,n3), ∂₂'*∂₂, sparse(I,n₁,n₁)) + 
           kron(del3'*del3, sparse(I,n₂,n₂), sparse(I,n₁,n₁))
end

"""
  punch_holes_3D(centers, radius, xpoints, ypoints, zpoints)

...
# Arguments

  - `centers::Vector{T}`: the vector containing the centers of the punches
  - `radius::Float64`: the radius of the punch
  - `Nx::Int64`: the number of unit cells in the x-direction, this code is hard-coded to start from one. 
  - `Ny::Int64`: the number of unit cells in the y-direction
  - `Nz::Int64`: the number of unit cells in the z-direction
...

...
# Outputs
  - `absolute_indices::Vector{Int64}`: vector containing the indices of coordinates 
  inside the punch
...

"""
function punch_holes_3D(centers, radius, Nx, Ny, Nz)
    clen = length(centers)
    masking_data_points = []
    absolute_indices = Int64[]
    for a = 1:clen
        c = centers[a]
        count = 1
        for i = 1:Nz
            for j = 1:Ny
                for h = 1:Nx
                    if((h-c[1])^2 + (j-c[2])^2 + (i - c[3])^2 <= radius^2)
                        append!(masking_data_points, [(h, j, i)])
                        append!(absolute_indices, count)
                    end
                    count = count +1
                end
            end
        end
    end
    return absolute_indices
end

#=
"""
  punch_holes_3D(centers, radius, xpoints, ypoints, zpoints)

...
# Arguments

  - `centers::Union{Vector,Matrix}`: the vector containing the centers of the punches
  - `radius::Tuple{Float64}`: the radii of the punch
  - `xpoints::Vector{T} where T<:Real`: the vector containing the x coordinate
  - `ypoints::Vector{T} where T<:Real`: the vector containing the y coordinate
  - `zpoints::Vector{T} where T<:Real`: the vector containing the z coordinate
...

...
# Outputs
  - `absolute_indices::Vector{Int64}`: vector containing the indices of coordinates 
  inside the punch
...

"""
function punch_holes_3D(centers, radius::Tuple{Float64}, 
                        xpoints::Vector, ypoints::Vector, zpoints::Vector)
    clen = (typeof(centers)<:Vector) ? length(centers) : 1
    masking_data_points = []
    absolute_indices = Int64[]
    for a = 1:clen
        c = (clen == 1) ? centers[a] : centers
        count = 1
        for i = 1:zpoints
            for j = 1:ypoints
                for h = 1:xpoints
                    if (((h-c[1])/radius[1])^2 + ((j-c[2])/radius[2])^2 + ((i - c[3])/radius[3])^2 <= 1)
                        append!(masking_data_points, [(h,j,i)])
                        append!(absolute_indices, count)
                            
                    end
                    count = count + 1
                end
            end
        end
    end
    return absolute_indices
end
=#

"""
  punch_holes_2D(centers, radius, xpoints, ypoints)

...
# Arguments

  - `centers::Vector{T} where T<:Real`: the vector containing the x coordinate
  - `radius::Vector{T} where T<:Real`: the vector containing the y coordinate
  - `xpoints::Vector{T} where T<:Real`: the vector containing the x coordinate
  - `ypoints::Vector{T} where T<:Real`: the vector containing the y coordinate
  - `zpoints::Vector{T} where T<:Real`: the vector containing the z coordinate
...

...
# Outputs
  - `absolute_indices::Vector{Int64}`: vector containing the indices of coordinates 
  inside the punch
...

"""
function punch_holes_2D(centers, radius, xpoints, ypoints)
    clen = length(centers)
    masking_data_points = []
    absolute_indices = Int64[]
    for a = 1:clen
        c = centers[a]
        count = 1
        for j = 1:ypoints
            for h = 1:xpoints
                if((h-c[1])^2 + (j-c[2])^2  <= radius^2)
                    append!(masking_data_points,[(h,j)])
                    append!(absolute_indices, count)
                end
                count = count + 1
            end
        end
        
    end
    return absolute_indices

end

"""
  punch_holes_2D(centers, radius, xpoints, ypoints)

...
# Arguments

  - `centers::Union{Vector, Matrix}`: the vector containing the punch centers
  - `radius::Vector`: the tuple containing the punch radii 
  - `xpoints::Vector{T} where T<:Real`: the vector containing the x coordinate
  - `ypoints::Vector{T} where T<:Real`: the vector containing the y coordinate
  - `zpoints::Vector{T} where T<:Real`: the vector containing the z coordinate
...

...
# Outputs
  - `absolute_indices::Vector{Int64}`: vector containing the indices of coordinates 
  inside the punch
...

"""
function punch_holes_2D(centers, radius::Tuple{Float64}, xpoints, ypoints)
    clen = length(centers)
    masking_data_points = []
    absolute_indices = Int64[]
    for a = 1:clen
        c = centers[a]
        count = 1
        for j = 1:ypoints
            for h = 1:xpoints
                if (((h-c[1])/radius[1])^2 + ((j-c[2])/radius[2])^2  <= 1.0)
                    append!(masking_data_points,[(h,j)])
                    append!(absolute_indices, count)
                end
                count = count + 1
            end
        end
    end
    return absolute_indices
end

"""
  Matern1D(h, N, f_array, args)

...
# Arguments
  - `h`: the
  - `N`: the number of data points
  - `f_array`: not sure
  - `args`: unclear
...

... 
# Outputs
  - matrix containing the Matern operator in one dimension.
...

"""
function Matern1D(h,N,f_array, args...)
    A= Tridiagonal([fill(-1.0/h^2, N-2); 0], [1.0; fill(2.0/h^2, N-2); 1.0], [0.0; fill(-1.0/h^2, N-2);])
    sizeA = size(A,1)
    epsilon = 2.2
    for i = 1:sizeA
        A[i,i] = A[i,i] + epsilon^2
    end
    A2 = A*A
    diag_c = ones(N)
    for i in discard
        diag_c[i] = 0
    end
    C = diagm(diag_c)
    Id = Matrix(1.0I, N, N)
    return (C-(Id -C)*A2)\(C*f_array)
end

"""

  Matern2D(xpoints, ypoints, imgg, epsilon, centers, radius, args...)

...
# Arguments
  - `xpoints::Vector{T} where T<:Real`: the vector containing the x coordinate
  - `ypoints::Vector{T} where T<:Real`: the vector containing the y coordinate
  - `imgg`: the matrix containing the image
  - `epsilon`: Matern parameter epsilon
  - `centers::Union{Vector, Matrix}`: the vector containing the punch centers
  - `radius::Vector`: the tuple containing the punch radii 
  - `args`: ?

# Outputs
  - tuple containing the restored image and the punched image, in grayscale.
...

"""
function Matern2D(xpoints, ypoints, imgg, epsilon, centers, radius, args...)
    A2D = ∇²(xpoints, ypoints)
    BoundaryNodes = return_boundary_nodes2D(xpoints, ypoints)
    for i in BoundaryNodes
        rowindices = A2D.rowval[nzrange(A2D, i)]
        A2D[rowindices,i].=0
        A2D[i,i] = 1.0
    end
    sizeA = size(A2D,1)
    for i = 1:sizeA
        A2D[i,i] = A2D[i,i] + epsilon^2
    end
    A2DMatern = A2D*A2D
    discard = punch_holes_2D(centers, radius, xpoints, ypoints)
    punched_image = copy(imgg)
    punched_image[discard] .= 1
    totalsize = prod(size(imgg))
    C = sparse(I, totalsize, totalsize)
    for i in discard
        C[i,i] = 0
    end
    Id = sparse(I, totalsize, totalsize)
    f = punched_image[:]
    rhs_a = C*f
    rhs_a = Float64.(rhs_a)
    u =((C-(Id -C)*A2DMatern)) \ rhs_a
    restored_img = reshape(u, xpoints, ypoints)
    restored_img = Gray.(restored_img)
    return (restored_img, punched_image)
end

"""

  Matern3D(xpoints, ypoints, zpoints, imgg, epsilon, centers, radius, args...)

...
# Arguments
  - `xpoints::Vector{T} where T<:Real`: the vector containing the x coordinate
  - `ypoints::Vector{T} where T<:Real`: the vector containing the y coordinate
  - `zpoints::Vector{T} where T<:Real`: the vector containing the z coordinate
  - `imgg`: the matrix containing the image
  - `epsilon`: Matern parameter epsilon
  - `centers::Union{Vector, Matrix}`: the vector containing the punch centers
  - `radius::Vector`: the tuple containing the punch radii 
  - `args`: ?

# Outputs
  - tuple containing the restored image and the punched image, in grayscale.
...

"""
function Matern3D(xpoints, ypoints, zpoints, imgg, epsilon, centers, radius, args...)
    A3D = ∇²3d(xpoints, ypoints, zpoints)
    BoundaryNodes = return_boundary_nodes(xpoints, ypoints, zpoints)
    for i in BoundaryNodes
        rowindices = A3D.rowval[nzrange(A3D, i)]
        A3D[rowindices,i] .= 0
        A3D[i,i] = 1.0
    end
    sizeA = size(A3D,1)
    for i = 1:sizeA
        A3D[i,i] = A3D[i,i] + epsilon^2
    end
    A3DMatern = A3D*A3D
    discard = punch_holes_3D(centers, radius, xpoints, ypoints, zpoints)
    punched_image = copy(imgg)
    punched_image[discard] .= 1
    totalsize = prod(size(imgg))
    C = sparse(I, totalsize, totalsize)
    for i in discard
        C[i,i] = 0
    end
    Id = sparse(I, totalsize, totalsize)
    f = punched_image[:]
    rhs_a = C*f
    rhs_a = Float64.(rhs_a)
    u = ((C-(Id -C)*A3DMatern)) \ rhs_a

    restored_img = reshape(u, xpoints, ypoints, zpoints)
    restored_img = Gray.(restored_img)
    return (restored_img, punched_image)
end

