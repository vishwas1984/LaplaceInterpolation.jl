

# Two-dimensional tests

@testset "Two dimensional interpolation" begin

    N = 4
    h = k = 1.0
    A2 = nablasq_2d_grid(N, N, h, k, 1)  #Neumann Boundaries
    @test (A2[1, 1] == 4.0) && (A2[2, 2] == 4.0) && (A2[6, 6] == 4.0)
    @test (A2[1, 2] == -2.0) && (A2[6, 5] == -2.0) && (A2[6, 9] == -1.0)
    
    A2 = nablasq_2d_grid(N, N, h, k, 0)  #Do Nothing Boundaries
    @test (A2[1, 1] == 2.0) && (A2[2, 2] == 3.0) && (A2[6, 6] == 4.0)

    x = y = 1:N
    h = k = Float64(x[2] - x[1])
    mat = sin.(2 * pi * x * 0.2) * cos.(2 * pi * y' * 0.3)
    discard = [1, 3, 6]
    # Laplace interpolation
    y_lap_N = matern_2d_grid(mat, discard, 1, 0.0, h, k, 1) # Neumann Boundaries
    y_lap_DN = matern_2d_grid(mat, discard, 1, 0.0, h, k, 0) # Do Nothing Boundaries
    # Matern interpolation
    y_mat_N = matern_2d_grid(mat, discard, 2, 0.01, h, k, 1) # Neumann Boundaries
    y_mat_DN = matern_2d_grid(mat, discard, 2, 0.01, h, k, 0) # Neumann Boundaries


    @test y_lap_N[discard] ≈ [-0.4755282581475768, 0.26582837761001243, -2.7755575615628914e-17]  
    @test y_lap_DN[discard] ≈ [-0.4755282581475768, 0.19592841743082434, -2.7755575615628914e-17]   
#    
    @test y_mat_N[discard] ≈ [-0.8634208541005728, 0.34933722092333386, -0.02321163330429045]
    @test y_mat_DN[discard] ≈ [-0.9831635001822346, 0.2957869652872276, 0.03934681698252432]

end

