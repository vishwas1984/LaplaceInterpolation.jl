
# One-dimensional tests

@testset "One dimensional interpolation" begin
    N = 4
    A1 = 2.0*sparse(I, N, N)
    A1[1, 1] = A1[4, 4] =  2.0
    A1[2, 1] = A1[3, 2] = A1[2, 3] = A1[3, 4] = -1.0
    A1[4, 3] = A1[1,2] = -2.0

    @test nablasq_1d_grid(N, 1.0, 1) == A1 # Neumann Boundary Conditions

    A1 = 2.0*sparse(I, N, N)
    A1[1, 1] = A1[4, 4] = 1.0 
    A1[2, 1] = A1[3, 2] = A1[2, 3] = A1[4, 3] = -1.0
    A1[3, 4] = A1[1,2] = -1.0

    @test nablasq_1d_grid(N,1.0, 0) == A1 # Do Nothing Boundary Conditions

    x = 1:N
    h = Float64(x[2] - x[1])
    y = sin.(2 * pi * x * 0.2)
    discard = [2, 4]
    # Laplace interpolation
    y_lap_N = matern_1d_grid(y, discard, 1, 0.0, h, 1) #Neumann Boundaries
    y_lap_DN = matern_1d_grid(y, discard, 1, 0.0, h, 0) #Do Nothing Boundaries
    # Matern interpolation
    y_mat_N = matern_1d_grid(y, discard, 2, 0.1, h, 1) #Neumann Boundaries
    y_mat_DN = matern_1d_grid(y, discard, 2, 0.1, h, 0) #Do Nothing Boundaries

    @test y_lap_N[discard] ≈ [0.18163563200134025, -0.587785252292473]
    @test y_lap_DN[discard] ≈ [0.18163563200134025, -0.2938926261462365]
    @test y_mat_N[discard] ≈ [0.334291319051756743, -0.8930938339654969]
    @test y_mat_DN[discard] ≈ [0.2503155527973194, -1.0026370054554663]

#@test y_mat[[2, 4]] ≈ [0.2503155527973194, -1.0026370054554663]
end

