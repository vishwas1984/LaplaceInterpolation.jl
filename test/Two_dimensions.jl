

# Two-dimensional tests

@testset "Two dimensional interpolation" begin

    N = 4
    h = k = 1.0
    A2 = nablasq_grid(N, N, h, k) 
    @test (A2[1, 1] == 2.0) && (A2[2, 2] == 3.0) && (A2[6, 6] == 4.0)

    x = y = 1:N
    h = k = Float64(x[2] - x[1])
    mat = sin.(2 * pi * x * 0.2) * cos.(2 * pi * y' * 0.3)
    discard = [1, 3, 6]
    # Laplace interpolation
    y_lap = matern_2d_grid(mat, discard, 1, 0.0, h, k)
    # Matern interpolation
    y_mat = matern_2d_grid(mat, discard, 2, 0.01, h, k)

    @test y_lap[discard] ≈ [-0.4755282581475768, 0.19592841743082434, -2.7755575615628914e-17]    
    @test y_mat[discard] ≈ [-0.9831635001822346, 0.2957869652872276, 0.03934681698252432] 

end

