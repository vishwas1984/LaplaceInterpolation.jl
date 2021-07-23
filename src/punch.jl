

"""
  punch_holes_2D(centers, radius, xpoints, ypoints)

...
# Arguments

  - `centers::Union{Vector}`: the vector containing the punch centers
  - `radius::Vector`: the tuple containing the punch radii 
  - `Nx::Int64`: the number of points in the x direction
  - `Ny::Int64`: the number of points in the y direction
...

...
# Outputs
  - `absolute_indices::Vector{Int64}`: vector containing the indices of coordinates 
  inside the punch
...

"""
function punch_holes_2D(centers, radius::Union{T, Vector{T}}, 
                        Nx, Ny) where T<:Number
    clen = (typeof(centers) <: Vector) ? length(centers) : 1
    rad  = (typeof(radius) <: Vector) ? radius : radius * ones(clen)
    masking_data_points = []
    absolute_indices = Int64[]
    for a = 1:clen
        c = centers[a]
        count = 1
        for j = 1:Ny
            for h = 1:Nx
                if (((h-c[1]))^2 + ((j-c[2]))^2  <= radius[a]^2)
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
  punch_holes_3D(centers, radius, xpoints, ypoints, zpoints)

...
# Arguments

  - `centers::Vector{T}`: the vector containing the centers of the punches
  - `radius::Float64`: the radius of the punch
  - `Nx::Int64`: the number of points in the x-direction, this code is hard-coded to start from one. 
  - `Ny::Int64`: the number of points in the y-direction
  - `Nz::Int64`: the number of points in the z-direction
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

"""
  punch_3D_cart(center, radius, xpoints, ypoints, zpoints; <kwargs>)

...
# Arguments

  - `center::Tuple{T}`: the tuple containing the center of a round punch
  - `radius::Union{Tuple{Float64},Float64}`: the radii/radius of the punch
  - `x::Vector{T} where T<:Real`: the vector containing the x coordinate
  - `y::Vector{T} where T<:Real`: the vector containing the y coordinate
  - `z::Vector{T} where T<:Real`: the vector containing the z coordinate

# Optional

  - `linear::Bool = false` can return a linear index
...

...
# Outputs
  - `inds::Vector{Int64}`: vector containing the indices of coordinates 
  inside the punch
...

"""
function punch_3D_cart(center, radius, x, y, z; linear = false)
    radius_x, radius_y, radius_z = (typeof(radius) <: Tuple) ? radius : 
                                                (radius, radius, radius)
    inds = filter(i -> (((x[i[1]]-center[1])/radius_x)^2 
                        + ((y[i[2]]-center[2])/radius_y)^2 
                        + ((z[i[3]] - center[3])/radius_z)^2 <= 1.0),
                  CartesianIndices((1:length(x), 1:length(y), 1:length(z))))
    (length(inds) == 0) && error("Empty punch.")
    if linear == false
      return inds
    else
      return LinearIndices(zeros(length(x), length(y), length(z)))[inds]
    end
end

