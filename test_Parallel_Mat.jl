
include("MaternKernelApproximation.jl")
(xmin, xmax) = (ymin, ymax) = (zmin,zmax) = (0.0, 1.0)
xpoints = ypoints = zpoints = -0.2:0.2:1.2
h = k = l = 0.2
imgg = randn(8,8,8)
m = 2
epsilon = 0.0
radius = (0.2, 0.3, 0.2)
restored = Parallel_Matern3D_Grid(xpoints, ypoints, zpoints, imgg, epsilon, radius, 
                                  h, k, l, xmin, xmax, ymin, ymax, zmin, zmax, m)
restored = Parallel_Laplace3D_Grid(xpoints, ypoints, zpoints, imgg, radius, 
                                  h, k, l, xmin, xmax, ymin, ymax, zmin, zmax)
