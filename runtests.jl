using Test, SparseArrays

# One-dimensional tests
N = 4
A1 = 2*matrix(I, N, N)
A1[1, 1] = A1[4, 4] = 1.0 
A1[2, 1] = A1[1, 2] = A1[3, 2] = A1[2, 3] = A1[4, 3] = A1[3, 4] = -1.0

@test nablasq_1d_grid(4, 1.0) == A1

x = 1:N
h = x[2] - x[1]
y = sin.(2 * pi * x * 0.2)
discard = [2, 4]
# Laplace interpolation
y_lap = matern_1d_grid(y, discard, 1, 0.0, h)
# Matern interpolation
y_mat = matern_1d_grid(y, discard, 2, 0.1, h)



