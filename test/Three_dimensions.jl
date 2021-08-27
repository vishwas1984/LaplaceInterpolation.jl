
# Three dimensional tests

@testset "Three dimensional interpolation" begin

    Nx = Ny = Nz = 4
    h = k = l = 1.0
    A3 = nablasq_3d_grid(Nx, Ny, Nz, h, k, l, 1)  # Neumann BC
    @test (A3[1, 1] == 6.0) && (A3[1, 2] == -2.0) && (A3[6, 6] == 6.0) && (A3[7, 9] == 0.0)

    A3 = nablasq_3d_grid(Nx, Ny, Nz, h, k, l, 0)  # Do Nothing BC
    @test (A3[1, 1] == 3.0) && (A3[2, 2] == 4.0) && (A3[6, 6] == 5.0) && (A3[22, 22] == 6.0)


    

    x = y = z = 1:Nx
    mat = [iz * (ix / 10) * cos.(pi * iy * 0.3) for ix in x for iy in y for iz in z]
    mat = reshape(mat, Nx, Ny, Nz)
    discard = [1, 2, 6, 7]
    # Laplace interpolation
    y_lap_N = matern_3d_grid(mat, discard, 1, 0.0, h, k, l, 1)  # Neumann BC
    y_lap_DN = matern_3d_grid(mat, discard, 1, 0.0, h, k, l, 0) # Do Nothing BC

    @test y_lap_N[discard] ≈ [0.058963784251626024, 0.0902360017338782, -0.08205577568503859, -0.11424405764762642]
    @test y_lap_DN[discard] ≈ [0.06272623215106933, 0.10152334543220813, -0.06808252702696804, -0.09721618012054391]
    
    # Matern interpolation
    discard = CartesianIndices(mat)[discard]
    y_mat_N = matern_3d_grid(mat, discard, 2, 0.01, h, k, l, 1) # Neumann BC
    y_mat_DN = matern_3d_grid(mat, discard, 2, 0.01, h, k, l, 0) # Do Nothing BC
    
#@test y_mat[discard] ≈ [0.09177622326365394, 0.14023883725830671, -0.0586259764263669, -0.09615104360848939]
    @test y_mat_N[discard] ≈ [0.10426268529646443, 0.15596454147863748, -0.057225005824516034, -0.09962579450169048]
    @test y_mat_DN[discard] ≈ [0.09177622326365394, 0.14023883725830671, -0.0586259764263669, -0.09615104360848939]

end
