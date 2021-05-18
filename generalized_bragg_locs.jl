# The idea behind this code is that we need to be able to give Bragg peaks as lists
# of indices before interpolating.

include("GeneralMK3D.jl")

N, M, K = (8, 8, 8)
imgg = randn(N, M, K)

(xmin, xmax) = (ymin, ymax) = (zmin,zmax) = (0.0, 1.0)
h = k = l = 0.2
xpoints = ypoints = zpoints = (xmin - h):h:(xmax + h)
imgg = randn(N, M, K)
m = 2
epsilon = 0.0
radius = (0.2, 0.3, 0.2)

centers = [(0.3, 0.3, 0.3), (0.8, 0.8, 0.8)]

discard = map(c -> punch_hole_3D(c, radius, xpoints, ypoints, zpoints), centers) 

res = copy(imgg)

# Threads.@threads for d in discard
for d in discard
    fi, li = (first(d), last(d) + CartesianIndex(1, 1, 1))
    selection = map(i -> i - fi + CartesianIndex(1, 1, 1), d)
    # Interpolate
    res[fi:li] = interp(xpoints[fi[1]:li[1]], ypoints[fi[2]:li[2]], 
            zpoints[fi[3]:li[3]], 
            imgg[fi:li], 
            selection, epsilon, m)
end
  
