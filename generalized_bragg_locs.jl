# The idea behind this code is that we need to be able to give Bragg peaks as lists
# of indices before interpolating.

include("MaternKernelApproximation.jl")

N, M, K = (8, 8, 8)
imgg = randn(N, M, K)
S = N*M*K

(xmin, xmax) = (ymin, ymax) = (zmin,zmax) = (0.0, 1.0)
xpoints = ypoints = zpoints = -0.2:0.2:1.2
h = k = l = 0.2
imgg = randn(8,8,8)
m = 2
epsilon = 0.0
radius = (0.2, 0.3, 0.2)

centers = [(0.3, 0.3, 0.3), (0.8, 0.8, 0.8)]

#=
mask = ones(size(imgg))
mask = broadcast(+, (xpoints .- c[1]).^2, (ypoints .- c[2])'.^2) .> rad^2

fun(c, k, img) = filter(x -> (floor(x / size(img,1) .- c[1]).^2 +  (x%size(img,1) .- c[2])'.^2 > rad^2, k)

indices = punch_holes_3D(centers, radius, xpoints, ypoints, zpoints)
=# 

mask = [(x - centers[1][1])^2 + (y - centers[1][2])^2 + (z - centers[1][3])^2 < 0.2^2 for x in xpoints for y in ypoints for z in zpoints]

