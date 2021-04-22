# This code interpolates for the missing points in an image. The code is specifically
# designed for removing Bragg peaks using the punch and fill algorithm. This code
# needs the image and the coordinates where the Bragg peaks needs to be removed and
# the radius (which can be the approximate width of the peaks). The code assumes all
# the "punches" will be of the same size and there are no Bragg peaks on the
# boundaries. Lines 2 to ~ 175 consists of helper functions and 175 onwards
# corresponds to the driver code.

# To do: - generalize the code so the aspect ratio can be different in h, k, l. 
#        - generalize the code so that m can be other than 2. 
#        - must make this code work when there is missing data on the boundary

using LinearAlgebra, SparseArrays, Colors

"""
  spdiagm_nonsquare(m, n, args)

Construct a sparse diagonal matrix from Pairs of vectors and diagonals. Each vector
arg.second will be placed on the arg.first diagonal. By default (if size=nothing), the
matrix is square and its size is inferred from kv, but a non-square size m×n (padded
with zeros as needed) can be specified by passing m,n as the first arguments.

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
  ∇²(n₁,n₂)

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
function ∇²3d_Grid(n₁,n₂,n3, h, k, l)
    o₁ = ones(n₁)/h
    ∂₁ = spdiagm_nonsquare(n₁+1,n₁,-1=>-o₁,0=>o₁)
    o₂ = ones(n₂)/k
    ∂₂ = spdiagm_nonsquare(n₂+1,n₂,-1=>-o₂,0=>o₂)
    O3 = ones(n3)/l
    del3 = spdiagm_nonsquare(n3+1,n3,-1=>-O3,0=>O3)
    return (kron(sparse(I,n3,n3),sparse(I,n₂,n₂), ∂₁'*∂₁) + 
            kron(sparse(I,n3,n3), ∂₂'*∂₂, sparse(I,n₁,n₁)) + 
            kron(del3'*del3, sparse(I,n₂,n₂), sparse(I,n₁,n₁)))
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
    BoundaryNodes3D =[];
    xneighbors = [];
    yneighbors = [];
    zneighbors = [];
    counter = 0;
    for k = 1:zpoints
        for j = 1:ypoints
            for i = 1:xpoints
                counter=counter+1;
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
    return BoundaryNodes3D, xneighbors, yneighbors,zneighbors
end

"""
  return_boundary_nodes2D(xpoints, ypoints)

...
# Arguments

  - `xpoints::Vector{T} where T<:Real`: the vector containing the x coordinate
  - `ypoints::Vector{T} where T<:Real`: the vector containing the y coordinate
...

...
# Outputs
  - `BoundaryNodes3D::Vector{Int64}`: vector containing the indices of coordinates 
  on the boundary of the 2D rectangle
...

"""
function return_boundary_nodes2D(xpoints, ypoints)
    BoundaryNodes2D =[];
    counter = 0;
    
    for j = 1:ypoints
        for i = 1:xpoints
            counter=counter+1;
            if( j == 1|| j == ypoints || i == 1 || i == xpoints)
                BoundaryNodes2D = push!(BoundaryNodes2D, counter)
            end
        end
    end
    
    return BoundaryNodes2D
end

"""
  punch_holes_nexus(xpoints, ypoints, zpoints, radius)

...
# Arguments

  - `xpoints::Vector{T} where T<:Real`: the vector containing the x coordinate
  - `ypoints::Vector{T} where T<:Real`: the vector containing the y coordinate
  - `zpoints::Vector{T} where T<:Real`: the vector containing the z coordinate
  - `radius::Float64`: the radius of the punch
...

...
# Outputs


  - `absolute_indices::Vector{Int64}`: vector containing the indices of coordinates 
  inside the punch

...
"""
# function punch_holes_nexus(xpoints, ypoints, zpoints, radius)
#     xlen = length(xpoints)
#     ylen = length(ypoints)
#     zlen = length(zpoints)
#     masking_data_points = []
#     absolute_indices = Int64[]
#     #Making sure that boundary is not punched
#     xmax = maximum(xpoints);
#     ymax = maximum(ypoints);
#     zmax = maximum(zpoints);
#     xmin = minimum(xpoints);
#     ymin = minimum(ypoints);
#     zmin = minimum(zpoints);
#     count = 1;

#     for i = 1:zlen
#         ir = round(zpoints[i])
#         for j = 1:ylen
#             jr = round(ypoints[j])
#             for h = 1:xlen
#                 hr = round(xpoints[h]);
#                 if(ir>=zmin+1 && ir <= zmax-1 && jr >= ymin+1 && jr <= ymax-1 && hr >= xmin+1 && hr <= xmax-1)
#                     if((hr-xpoints[h])^2 + (jr-ypoints[j])^2/(1.5^2) + (ir - zpoints[i])^2/(1.5^2) <= radius^2)
#                         #imgg_copy[h,j,i] = 1
#                         #append!(masking_data_points,[(h,j,i)]);
#                         append!(absolute_indices, count);   
#                     end
#                 end
#                 count = count + 1
#             end
#         end
#     end
#     return absolute_indices
# end

function punch_holes_nexus(xpoints, ypoints, zpoints, radius)
    xlen = length(xpoints)
    ylen = length(ypoints)
    zlen = length(zpoints)
    masking_data_points = []
    absolute_indices = Int64[]
    #Making sure that boundary is not punched
    
    count = 1;

    for i = 1:zlen
        ir = round(zpoints[i])
        for j = 1:ylen
            jr = round(ypoints[j])
            for h = 1:xlen
                hr = round(xpoints[h]);
                
                if((hr-xpoints[h])^2 + (jr-ypoints[j])^2/(1.5^2) + (ir - zpoints[i])^2/(1.5^2) <= radius^2)
                        #imgg_copy[h,j,i] = 1
                        #append!(masking_data_points,[(h,j,i)]);
                    append!(absolute_indices, count);   
                end
                
                count = count + 1
            end
        end
    end
    return absolute_indices
end

"""
  punch_holes_3D(centers, radius, xpoints, ypoints, zpoints)

...
# Arguments

  - `centers::Vector{T}`: the vector containing the centers of the punches
  - `radius::Float64`: the radius of the punch
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
function punch_holes_3D(centers, radius, xpoints, ypoints, zpoints)
    clen = length(centers)
    masking_data_points = []
    absolute_indices = Int64[]
    for a = 1:clen
        c=centers[a]
        count = 1
        for i = 1:zpoints
            for j = 1:ypoints
                for h = 1:xpoints
                    if((h-c[1])^2 + (j-c[2])^2 + (i - c[3])^2 <= radius^2)
                        append!(masking_data_points,[(h,j,i)])
                        append!(absolute_indices, count)
                    end
                    count = count +1
                end
            end
        end
    end
    return absolute_indices
end

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
    clen = length(centers);
    masking_data_points = [];
    absolute_indices = Int64[];
    for a = 1:clen
        c=centers[a];
       
        count = 1;
        
        for j = 1:ypoints
            for h = 1:xpoints
                if((h-c[1])^2 + (j-c[2])^2  <= radius^2)
                    #imgg_copy[h,j,i] = 1
                    append!(masking_data_points,[(h,j)]);
                    append!(absolute_indices, count);
                        
                end
                count = count +1;
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

"""

  Matern3D_grid(xpoints, ypoints, zpoints, imgg, epsilon, radius, h, k, l, args...)

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
  - `m::Int` : Matern parameter (use 2 for Matern Kernel)
  - `args`: ?

# Outputs
  - tuple containing the restored image and the punched image.
...

"""
function Matern3D_Grid(xpoints, ypoints, zpoints, imgg, epsilon, radius, h, k, l, m, args...)
    xlen = length(xpoints)
    ylen = length(ypoints)
    zlen = length(zpoints)
    A3D = ∇²3d_Grid(xlen, ylen, zlen, h, k, l)
    BoundaryNodes, xneighbors, yneighbors, zneighbors = return_boundary_nodes(xlen, ylen, zlen)
    # for i in BoundaryNodes
    #     rowindices = A3D.rowval[nzrange(A3D, i)]
    #     A3D[rowindices,i].=0
    #     A3D[i,i] = 1.0
    # end
    count = 1;
    for i in BoundaryNodes
        A3D[i,i] = 0.0;
        A3D[i,i] = A3D[i,i] + xneighbors[count]/h^2 + yneighbors[count]/k^2 + zneighbors[count]/l^2;
        count = count + 1;
    end

    sizeA = size(A3D,1)
    for i = 1:sizeA
        A3D[i,i] = A3D[i,i] + epsilon^2
    end
    A3DMatern = A3D;
    for i = 1:m-1
        A3DMatern = A3DMatern*A3D;
    end
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
    u = ((C-(Id -C)*A3DMatern)) \ rhs_a
    return (u, punched_image[:])
end

"""

  Laplace3D_grid(xpoints, ypoints, zpoints, imgg, radius, h, k, l, args...)

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
  - `args`: ?

# Outputs
  - tuple containing the restored image and the punched image.
...

"""

function Laplace3D_Grid(xpoints, ypoints, zpoints, imgg, radius, h, k, l, args...)
    xlen = length(xpoints)
    ylen = length(ypoints)
    zlen = length(zpoints)
    A3D = ∇²3d_Grid(xlen, ylen, zlen, h, k, l)
    #BoundaryNodes = return_boundary_nodes(xlen, ylen, zlen)
    BoundaryNodes, xneighbors, yneighbors, zneighbors = return_boundary_nodes(xlen, ylen, zlen)
    count = 1;
    for i in BoundaryNodes
        A3D[i,i] = 0.0;
        A3D[i,i] = A3D[i,i] + xneighbors[count]/h^2 + yneighbors[count]/k^2 + zneighbors[count]/l^2;
        count = count + 1;
    end

    # for i in BoundaryNodes
    #     rowindices = A3D.rowval[nzrange(A3D, i)]
    #     A3D[rowindices,i] .= 0
    #     A3D[i,i] = 1.0
    # end
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
    u =((C-(Id -C)*A3D)) \ rhs_a
    return u, punched_image[:]
end

