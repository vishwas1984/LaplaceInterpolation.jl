using Laplacians, SparseArrays, LinearAlgebra

function laplacians_2d(a_num,S,vals)
    a = Laplacians.grid2(reverse(a_num)...)
    S .+= 1
    x = Laplacians.harmonic_interp(a, S, vals)
    x = reshape(x,a_num...)
    return x
end

function laplacians_julia(a_num,S,vals)
    a = Laplacians.grid3(reverse(a_num)...)
    S .+= 1
    x = Laplacians.harmonic_interp(a, S, vals)
    x = reshape(x,a_num...)
    return x
end

"""
Matern interpolation using the laplacian form
"""
function matern_interp(a, S::Vector, vals::Vector, epsilon, m; tol=1e-6)
    n = size(a,1)
    b = zeros(n)
    b[S] = vals

    inds = ones(Bool,n)
    inds[S] .= false
    la = (lap(a) - epsilon^2*sparse(I, n, n))^m

    la_sub = la[inds,inds]
    b_sub = (-la*b)[inds]
    # f = chol_sddm(la_sub; tol=tol)
    f = cgSolver(la_sub; tol=tol)
    x_sub = f(b_sub)

    x = copy(b)
    x[inds] = x_sub
    return x
end

function maternDE_julia(a_num, S, vals, epsilon, m)
    a = Laplacians.grid3(reverse(a_num)...)
    S .+= 1
    x = matern_interp(a, S, vals, epsilon, m)
    x = reshape(x,a_num...)
    return x
end

function matern2D_julia(a_num, S, vals, epsilon, m)
    a = Laplacians.grid3(reverse(a_num)...)
    S .+= 1
    x = Laplacians.harmonic_interp(a, S, vals, epsilon, m)
    x = reshape(x,a_num...)
    return x
end