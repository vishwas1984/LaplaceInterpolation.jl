module LaplaceInterpolation

  using LinearAlgebra, SparseArrays

  include("Matern1D2D.jl")
  export nablasq_grid, bdy_nodes, matern_1d_grid, matern_2d_grid 

  include("GeneralMK3D.jl")
  export nablasq_3d_grid, matern_3d_grid, Laplace_3D_Grid, parallel_Matern_3DGrid
  export parallel_Laplace_3Dgrid, interp_nexus 

  include("MaternKernelApproximation.jl")
  export spdiagm_nonsquare, return_boundary_nodes, Matern3D_Grid, Parallel_Matern3D_Grid

  include("punch.jl")
  export punch_holes_nexus, punch_holes_3D, punch_holes_2D

end
