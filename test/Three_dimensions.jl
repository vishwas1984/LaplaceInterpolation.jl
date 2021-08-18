
# Three dimensional tests

@testset "Three dimensional interpolation" begin

    Nx = Ny = Nz = 4
    h = k = l = 1.0
    A3 = nablasq_3d_grid(Nx, Ny, Nz, h, k, l) 
    @test (A3[1, 1] == 6.0) && (A3[2, 2] == 6.0) && (A3[6, 6] == 6.0) && (A3[7, 9] == 0.0)

    x = y = z = 1:Nx
    mat = [iz * (ix / 10) * cos.(pi * iy * 0.3) for ix in x for iy in y for iz in z]
    mat = reshape(mat, Nx, Ny, Nz)
    discard = [1, 2, 6, 7]
    # Laplace interpolation
    y_lap = matern_3d_grid(mat, discard, 1, 0.0, h, k, l)

#@test y_lap[discard] ≈ [0.06272623215106933, 0.10152334543220813, -0.06808252702696804, -0.09721618012054391]
    @test y_lap[discard] ≈ [0.02488766206545813, 0.06267311013795539, -0.06029241053144326, -0.0797138021211637]
    
    # Matern interpolation
    discard = CartesianIndices(mat)[discard]
    y_mat = matern_3d_grid(mat, discard, 2, 0.01, h, k, l)
#@test y_mat[discard] ≈ [0.09177622326365394, 0.14023883725830671, -0.0586259764263669, -0.09615104360848939]
    @test y_mat[discard] ≈ [0.04421500582500884, 0.09755480304997476, -0.05903324928597118, -0.08583301152639768]

end
