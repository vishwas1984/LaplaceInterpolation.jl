using Laplacians, LinearAlgebra, SparseArrays
using TestImages, Colors, Plots, FileIO, BenchmarkTools

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

include("laplacians_julia.jl")

# function matern_interp(a, S::Vector, vals::Vector, epsilon, m; tol=1e-6)
#     n = size(a,1)
#     b = zeros(n)
#     b[S] = vals

#     inds = ones(Bool,n)
#     inds[S] .= false
#     la = (lap(a) + epsilon^2*sparse(I, n, n))^m

#     la_sub = la[inds,inds]
#     b_sub = (-la*b)[inds]
#     f = cgSolver(la_sub; tol=tol)
#     x_sub = f(b_sub)

#     x = copy(b)
#     x[inds] = x_sub
#     return x
# end

function matern_wrapper(center, xpoints, ypoints, radius, mat, epsilon, m)
    absolute_indices = punch_holes_2D(center, radius, xpoints, ypoints);
    all_indices = 1:xpoints*ypoints;
    known_indices = setdiff(all_indices, absolute_indices); 
    a = grid2(ypoints, xpoints);
    vals = mat[known_indices];
    y = matern_interp(a, known_indices, vals, epsilon, 2, tol=1e-6);
    y = reshape(y,xpoints,ypoints);
    return y
end


img = testimage("mandrill");

imgg = Gray.(img);

mat = convert(Array{Float64}, imgg)[1:256,1:512];
# This image is square
cent = [(100, 200), (200, 100), (200, 400)]
radius = 20;
xpoints = size(mat,1);
ypoints = size(mat,2);
y = matern_wrapper(cent, xpoints, ypoints, radius, mat, 0.2, 2)
y = reshape(y, xpoints, ypoints);
plot(Gray.(y))
# BenchmarkTools.DEFAULT_PARAMETERS.seconds = 600;
# BenchmarkTools.DEFAULT_PARAMETERS.samples = 50;
# @benchmark matern_wrapper(cent, xpoints, ypoints, radius, mat, 0.2, 2)
# absolute_indices = punch_holes_2D(cent, radius, xpoints, ypoints);
