

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
                        xpoints, ypoints) where T<:Number
    clen = (typeof(centers) <: Vector) ? length(centers) : 1
    rad  = (typeof(radius) <: Vector) ? radius : radius * ones(clen)
    masking_data_points = []
    absolute_indices = Int64[]
    for a = 1:clen
        c = centers[a]
        count = 1
        for j = 1:ypoints
            for h = 1:xpoints
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

