module LaplaceInterpolation

  using LinearAlgebra, SparseArrays

  include("Matern1D2D")
  export ∇²1d_Grid, ∇²2d_Grid, Matern_1D_Interplation, Matern1D, Matern2D 

  include("GeneralMK3D.jl")
  export ∇²3d_Grid, Matern_3D_Grid, Laplace_3D_Grid, parallel_Matern_3DGrid
  export parallel_Laplace_3Dgrid 

end
