
# This script will allow one to run the notebooks from the REPL
# IJulia.jl is required
# First, navigate to the Notebooks directory in laplaceinterpolation and issue 
#  
# Then run this script from the REPL

using Pkg, IJulia

Pkg.activate(".")
Pkg.instantiate()
notebook(dir=".")

