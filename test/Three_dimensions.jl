
# Three dimensional tests

@testset "Three dimensional interpolation" begin

    Nx = Ny = Nz = 4
    h = k = l = 1.0
    A3 = nablasq_3d_grid(Nx, Ny, Nz, h, k, l) 
    @test (A3[1, 1] == 3.0) && (A3[2, 2] == 4.0) && (A3[6, 6] == 5.0) && (A3[7, 9] == -1.0)

    x = y = z = 1:Nx
    mat = [iz * (ix / 10) * cos.(pi * iy * 0.3) for ix in x for iy in y for iz in z]
    discard = [1, 2, 6, 7]
    # Laplace interpolation
    y_lap = matern_3d_grid(mat, discard, 1, 0.0, h, k, l)
    # Matern interpolation
    y_mat = matern_3d_grid(mat, discard, 2, 0.01, h, k, l)

    # @test y_lap[discard] ≈ [-0.4755282581475768, 0.19592841743082434, -2.7755575615628914e-17]    
    # @test y_mat[discard] ≈ [-0.9831635001822346, 0.2957869652872276, 0.03934681698252432] 

end
