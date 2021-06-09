# Laplace and Matern Interpolation for Volume Datasets
This code performs Laplace and Matern interpolation for volume datasets. The code is especially useful for studying crystal structures which contain Bragg peaks.  ```MaternKernelApproximation.jl``` takes in the volume data and uses a punch and fill algorithm to remove the Bragg peaks and interpolates for missing values at punch locations. We provide two options for interpolation: Laplace interpolation and Matern Interpolation. 

# Dependencies
```MaternKernelApproximation.jl``` itself requires only ```LinearAlgebra``` and ```SparseArrays``` packages. A Jupyter notebook that illustrates the use of ```MaternKernelApproximation.jl``` can be found at ```Notebooks/MaternInterpolationWorkflow.ipynb```. The jupyter notebook illustrates the usage for smoothing out Bragg peaks in Molybdenum Vanadium Dioxide dataset. 


Bragg Peaks                | Matern and Laplace Interpolation 
:-------------------------:|:--------------------------------:
![](Slides/BragPeaks.png)  |  ![](Slides/Punch_Fill.png)

# Funding
This project is funded by the US DOE under the BES program.