
# see readme for module documentation
module PyAMG

export RugeStubenSolver,
       SmoothedAggregationSolver,
       solve, solve!, aspreconditioner,
       set_kwargs!


using LinearAlgebra
using PyCall
using SparseArrays

pyamg = PyNULL()
scipy_sparse = PyNULL()
function __init__()
    copy!(pyamg, pyimport_conda("pyamg", "pyamg"))
    copy!(scipy_sparse, pyimport_conda("scipy.sparse", "scipy"))
end


"""
`py_csc(A::SparseMatrixCSC) -> PyObject`

Takes a Julia CSC matrix and converts it into `PyObject`, which stores a
`scipy.sparse.csc_matrix`.
"""
function py_csc(A::SparseMatrixCSC)
   # create an empty sparse matrix in Python
   Apy = scipy_sparse.csc_matrix(size(A))
   # write the values and indices
   Apy.data = copy(A.nzval)
   Apy.indices = A.rowval .- 1
   Apy.indptr = A.colptr .- 1
   return Apy
end


"""
`py_csr(A::SparseMatrixCSC) -> PyObject`

Takes a Julia CSC matrix and converts it into `PyObject`, which stored a
`scipy.sparse.csr_matrix`. (Note: it first converts it to `csc_matrix`, then
calls `tocsr()`)

TODO: this seems extremely inefficient and should at some point be fixed.
"""
py_csr(A::SparseMatrixCSC) = py_csc(A).tocsr()


"""
`struct AMGSolver{T}`

Encapsulates the AMG solver types implemented in PyAMG.

Initialise using
```
using PyAMG
amg_rs = RugeStubenSolver(A)
amg_sa = SmoothedAggregationSolver(A)
```
To solve Ax = b:
```
x = solve(amg, b, tol=1e-6, accel="cg")
```
"""
mutable struct AMGSolver{T}
    po::PyObject
    id::T
    kwargs::Vector
    A::SparseMatrixCSC
end

struct RugeStuben end
struct SmoothedAggregation end
global const RugeStubenSolver = AMGSolver{RugeStuben}
global const SmoothedAggregationSolver = AMGSolver{SmoothedAggregation}


function set_kwargs!(amg::AMGSolver; kwargs...)
   amg.kwargs = collect(kwargs)
end



"""
`RugeStubenSolver(A::SparseMatrixCSC, kwargs...)`:

Create a Ruge Stuben instance of `AMGSolver`; wraps
`pyamg.ruge_stuben_solver`; see `pyamg.ruge_stuben_solver?`
for keyword arguments.  See `?AMGSolver` for usage.
"""
RugeStubenSolver(A::SparseMatrixCSC; kwargs...) =
   AMGSolver(pyamg.ruge_stuben_solver(py_csr(A)),
             RugeStuben(), collect(Any, kwargs), A)


"""
`SmoothedAggregationSolver(A::SparseMatrixCSC, kwargs...)`

Wrapper for `pyamg.smoothed_aggregation_solver`; see `pyamg.ruge_stuben_solver?`
for keyword arguments. See `?AMGSolver` for usage.
"""
SmoothedAggregationSolver(A::SparseMatrixCSC; kwargs...) =
   AMGSolver(pyamg.smoothed_aggregation_solver(py_csr(A)),
             SmoothedAggregation(), collect(Any, kwargs), A)


"""
`solve(A::SparseMatrixCSC, b::Vector; kwargs...)`:

PyAMG's 'blackbox' solver. See `pyamg.solve?` for `kwargs`.
"""
function solve(A::SparseMatrixCSC, b::Vector; kwargs...)
   # If kwargs contains :residuals, we need to do some conversions, since
   # Python cannot append to Julia arrays (i.e. numpy arrays).
   for (n, (key, rj)) in enumerate(kwargs)
      if key == :residuals
         rp = PyVector(Float64[])
         kwargs[n] = (:residuals, rp)
         try
            x = pyamg.solve( py_csr(A), b; kwargs... )
            append!(rj, collect(rp))
            return x::Vector{Float64}
         catch
            error("Something went wrong. Probably, your version of pyamg probably does not support the `residuals` keyword; please update `pyamg` (see https://github.com/pyamg/pyamg) or call `PyAMG.solve` without a `residuals` keyword.")
         end
      end
   end
   # If we are here, then we just solve and return
   return pyamg.solve( py_csr(A), b; kwargs... )::Vector{Float64}
end


"""
`solve(amg::AMGSolver, b, kwargs...)`

Returns a `Vector` with the result of the AMG solver. The keyword
arguments can either be passed directly, or can be stored in
`amd` via `set_kwargs!`.


### `kwargs`  (copy-pasted from Python docs)

* `x0` : Initial guess.
* `tol` : Stopping criteria: relative residual r[k]/r[0] tolerance.
* `maxiter` : Stopping criteria: maximum number of allowable iterations.
* `cycle` : {"V","W","F","AMLI"}
    Type of multigrid cycle to perform in each iteration.
* `accel` : Defines acceleration method.  Can be a string such as "cg"
    or "gmres" which is the name of an iterative solver in
    `pyamg.krylov` (preferred) or scipy.sparse.linalg.isolve.
    If accel is not a string, it will be treated like a function
    with the same interface provided by the iterative solvers in SciPy.
         (the function version is not tested in Julia!)
* `callback` : User-defined function called after each iteration.  It is
    called as callback(xk) where xk is the k-th iterate vector.
         (also not tested in Julia!)
* `residuals` : List to contain residual norms at each iteration.
"""
function solve(amg::AMGSolver, b::Vector; kwargs...)
   # If kwargs contains :residuals, we need to do some conversions, since
   # Python cannot append to Julia arrays (i.e. numpy arrays).
   if haskey(kwargs, :residuals)
      rp = PyVector(Float64[])
      x = amg.po.solve(b; amg.kwargs..., kwargs..., residuals=rp)
      append!(kwargs[:residuals], collect(rp))
      return x::Vector{Float64}
   end
   # If we are here, then we just solve and return
   return amg.po.solve(b; amg.kwargs..., kwargs...)::Vector{Float64}
end


# function solve(amg::AMGSolver, b; history=false, kwargs...)
#    if history
#       r = PyVector(Float64[])
#       x = amg.po.solve(b; amg.kwargs..., kwargs..., residuals=r)
#       return x, collect(r)
#    else
#       return amg.po.solve(b; amg.kwargs..., kwargs...)
#    end
# end



######### Capability to use PyAMG.jl as a preconditioner for
######### nonlinear optimisation, sampling, etc

# TODO: the following 4 methods still need to be tested

import Base.\, Base.*
\(amg::AMGSolver, b::Vector) = solve(amg, b; amg.kwargs...)
*(amg::AMGSolver, x::Vector) = amg.A * x

LinearAlgebra.ldiv!(x, amg::AMGSolver, b) = copyto!(x, amg \ b)
LinearAlgebra.mul!(b, amg::AMGSolver, x) = mul!(b, amg.A, x)


##############################################################################
######### Capability to use PyAMG.jl as a preconditioner for
######### iterative linear algebra

"""
`struct AMGPreconditioner`

returned by `aspreconditioner(amg)`, when `amg` is of type `AMGSolver`.
This stores `PyObject` that acts as a linear operator. This type
should be used when `PyAMG` is used as a preconditioner for
iterative linear algebra.

Overloaded methods that can be used with an `AMGPreconditioner` are
`\`, `*`
"""
struct AMGPreconditioner
  po::PyObject
  A::SparseMatrixCSC
end

"""
`aspreconditioner(amg::AMGSolver; kwargs=...)`

returns an `M::AMGPreconditioner` object that is suitable for usage
as a preconditioner for iterative linear algebra.

If `x` is a vector, then `M \\ x` denotes application of the
preconditioner (i.e. 1 MG cycle), while `M * x` denotes
multiplication with the original matrix from which `amg` was constructed.

### kwargs:
cycle : {'V','W','F','AMLI'}
    Type of multigrid cycle to perform in each iteration.
"""
aspreconditioner(amg::AMGSolver; kwargs...) =
      AMGPreconditioner(amg.po.aspreconditioner(kwargs...), amg.A)

\(amg::AMGPreconditioner, b::Vector) = amg.po.matvec(b)
*(amg::AMGPreconditioner, x::Vector) = amg.A * x

LinearAlgebra.ldiv!(x, amg::AMGPreconditioner, b) = copyto!(x, amg \ b)
LinearAlgebra.mul!(b, amg::AMGPreconditioner, x) = mul!(b, amg.A, x)



"""
`diagnostics(A)`:

Wrapper for `solver_diagnostics`, which is part of PyAMG-Examples. To use it
clone [PyAMG-Examples](https://github.com/pyamg/pyamg-examples), then make the
call to `diagnostics` from a directory where `solver_diagnostics.py` is located
(e.g., from `pyamg-examples/solver_diagnostics`, but the file could be copied
anywhere.

If succesful, `diagnostics(A)` will try a variety of parameter combinations for
`smoothed_aggregation_solver`, and write two files 'solver_diagnostic.txt' and
'solver_diagnostic.py' which contain information how to generate the best
solver.

This implementation is a really poor hack, and suggestions how to improve it
would be highly appreciated.
"""
function diagnostics(A::SparseMatrixCSC; kwargs...)
    # try to import solver_diagnostics
    try
        unshift!(PyVector(pyimport("sys")["path"]), "");
        solver_diagnostics = pyimport("solver_diagnostics")
    catch
        error("""
              I tried to pyimport solver_diagnostics, but it fails.
              This is probably because `solver_diagnostics.py` is not
              in the current directory. Please see
              `?PyAMG.diagonistics` on how to use this function.
              """)
    end

    # import has worked against expectations; call the diagnostics
    solver_diagnostics.solver_diagnostics(A; kwargs...)
end


end
