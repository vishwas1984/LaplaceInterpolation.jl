
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
function return_boundary_nodes2D(xpoints, ypoints)
  BoundaryNodes2D = []
  xneighbors = []
  yneighbors = []
  counter = 0
      for j = 1:ypoints
          for i = 1:xpoints
              counter = counter + 1
              if(j == 1|| j == ypoints || i == 1 || i == xpoints)
                  BoundaryNodes2D = push!(BoundaryNodes2D, counter)
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
  return BoundaryNodes2D, xneighbors, yneighbors
end

""" laplacian matrix on a 1D grid"""
function ∇²1d_Grid(n₁, h)
  o₁ = ones(n₁)/h
  ∂₁ = spdiagm_nonsquare(n₁+1,n₁,-1=>-o₁,0=>o₁)
  A1D = ∂₁'*∂₁
  A1D[1,1] = 1.0/h^2
  A1D[n₁, n₁] = 1.0/h^2
  return A1D
end

""" Matern Interpolation in one dimension"""
function Matern_1D_Interpolation(n1, h, missing_data_index, m, epsilon, known_values)
  A1D = ∇²1d_Grid(n1, h)
  dimension = 1
  C = sparse(I, len, len)
  Id = sparse(I, len, len);
  for i in missing_data_index
    C[i,i] = 0;
  end
  u =((C-(Id -C)*A1D)) \ (known_values);
  laplace_interpolated_data = u
  for i = 1:size(A1D,1)
    A1D[i,i] = A1D[i,i] + epsilon^2
  end

  A1DM = A1D^m
  u =((C-(Id -C)*A1DM)) \ (known_values);
  matern_interpolated_data = u
  return laplace_interpolated_data, matern_interpolated_data
end

""" 
   Matern_1D_Grid(y, h, missing_data_index, m, epsilon)

Matern Interpolation in one dimension

# Arguments:
  -`y`:data
  -`h`:aspect ratio
  -`missing_data_index` indices of missing values
  - `m` matern parameter
  - `epsilon` other matern parameter.

"""
function Matern_1D_Grid(y, h, missing_data_index, m, epsilon)
  A1D = ∇²1d_Grid(length(y), h)
  len = size(A1D, 1)
  C = sparse(I, len, len)
  Id = sparse(I, len, len)
  for i in missing_data_index
    C[i,i] = 0
  end
  known_values = C*y
  if ((m == 1)||(m == 1.0)) && (epsilon == 0.0)
    return ((C - (Id - C) * A1D)) \ (known_values)
  else
    for i = 1:size(A1D,1)
      A1D[i,i] = A1D[i,i] + epsilon^2
    end
    A1DM = A1D^m
    return ((C-(Id -C)*A1DM)) \ (known_values)
  end
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

""" 2D laplacian on a grid """
function ∇²2d_Grid(n₁, n₂, h, k)
  o₁ = ones(n₁) / h
  ∂₁ = spdiagm_nonsquare(n₁ + 1, n₁, -1 => -o₁, 0 => o₁)
  o₂ = ones(n₂) / k
  ∂₂ = spdiagm_nonsquare(n₂ + 1, n₂, -1 => -o₂,0 => o₂)
  # O3 = ones(n3) / l
  # del3 = spdiagm_nonsquare(n3 + 1, n3, -1 => -O3, 0 => O3)
  A2D = (kron(sparse(I, n₂, n₂), ∂₁'*∂₁) + 
          kron(∂₂' * ∂₂, sparse(I, n₁, n₁)))
  BoundaryNodes, xneighbors, yneighbors = 
          return_boundary_nodes2D(n₁, n₂)
  count = 1
  for i in BoundaryNodes
      A2D[i, i] = 0.0
      A2D[i, i] = A2D[i, i] + xneighbors[count] / h ^ 2 + yneighbors[count] / k ^ 2 
      count = count + 1
  end
  return A2D
end

#=
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
=#

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
    A2D = ∇²2d_Grid(xpoints, ypoints,1,1)
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
    return (restored_img, punched_image)
end

"""

  Matern2D_Grid(xpoints, ypoints, imgg, epsilon, m)

...
# Arguments
  - `mat`: the matrix containing the image
  - `epsilon`: Matern parameter epsilon
  - `m`: The Matern exponent (integer)
  - `discard`: the linear indices of the nodes to be discarded

# Outputs
  - tuple containing the interpolated image
...

"""
function Matern2D_Grid(mat, epsilon, m, discard)
    rows, columns = size(mat)
    A2D = ∇²2d_Grid(rows, columns, 1, 1)
    C = sparse(I, rows * columns, rows * columns)
    for i in discard
        C[i,i] = 0
    end
    Id = sparse(I, rows * columns, rows * columns)
    f = mat[:]
    if (m == 0)||(m == 0.0)
        # Laplace interpolation
        u = ((C - (Id - C) * A2D)) \ (C*f)
        return reshape(u, rows, columns)
    else
        sizeA = size(A2D, 1)
        for i = 1:sizeA
            A2D[i,i] = A2D[i,i] + epsilon^2
        end
        A2D = A2D^m
        u = ((C - (Id - C) * A2D)) \ (C * f)
        return reshape(u, rows, columns)   
    end
end


#=
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
=#
