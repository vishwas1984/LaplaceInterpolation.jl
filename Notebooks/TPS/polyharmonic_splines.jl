
#polyharmonic_splines.jl
# See: https://github.com/lstagner/PolyharmonicSplines.jl
#=
The MIT License (MIT)
Copyright (c) 2015 Luke Stagner
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
=#

struct PolyharmonicSpline
    dim::Int64
    order::Int64
    coeff::Vector{Float64}
    centers::Array{Float64,2}
    error::Float64
end

function polyharmonicK(r,K)
    if iseven(K)
        iszero(r) && return zero(r)
        if r >= 1.0
            return (r^K)*log(r)
        elseif 0.0 < r < 1.0
            return (r.^(K-1))*log(r.^r)
        elseif iszero(r) # Needed for autodiff to work at zero
            return zero(r)
        end
    else
        return r^K
    end
end

function PolyharmonicSpline(K::Int64, centers::Array{Float64,2}, values::Array{Float64}; s = 0.0)
    m,n = size(centers)
    m != length(values) && throw(DimensionMismatch())

    M = zeros(m,m)
    N = zeros(m,n+1)

    for i=1:m
        N[i,1] = 1
        N[i,2:end] = centers[i,:]
        for j=1:m
            M[i,j] = polyharmonicK(norm(centers[i,:] .- centers[j,:]),K)
        end
    end
    M = M + s*I
    L = vcat(hcat(M,N),hcat(N', zeros(n+1,n+1)))

    w = pinv(L)*vcat(values,zeros(n+1))

    ivalues = zeros(m)
    for i=1:m
        tmp = 0.0
        for j=1:m
            tmp = tmp + w[j]*polyharmonicK(norm(centers[i,:] .- centers[j,:]),K)
        end
        tmp = tmp + w[m+1]
        for j=2:n+1
            tmp = tmp + w[m+j]*centers[i,j-1]
        end
        ivalues[i] = tmp
    end
    error = norm(values .- ivalues)

    return PolyharmonicSpline(n,K,w,centers,error)
end

function PolyharmonicSpline(K::Int64, centers::Vector{Float64},values::Vector{Float64};s = 0.0)
    PolyharmonicSpline(K,reshape(centers,length(centers),1),values,s=s)
end

function (S::PolyharmonicSpline)(x::T...) where T <: Real
    n = length(x)
    n != S.dim && throw(DimensionMismatch("$n != $(S.dim)"))

    v = 0.0
    l = length(S.coeff)-(n+1)
    for j=1:l
        v = v + S.coeff[j]*polyharmonicK(norm(x .- S.centers[j,:]), S.order)
    end

    v = v + S.coeff[l+1]
    for j=2:n+1
        v = v + S.coeff[l+j]*x[j-1]
    end

    return v
end

function interpolate(S::PolyharmonicSpline,x::Array{Float64,2})
    m,n = size(x)
  
    n != S.dim && throw(DimensionMismatch("$m != $(S.dim)"))
  
    interpolates = zeros(m)
    for i=1:m
      tmp = 0.0
      l = length(S.coeff)-(n+1)
      for j=1:l
        tmp = tmp + S.coeff[j]*polyharmonicK(norm(x[i,:] .- S.centers[j,:]),S.order)
      end
      tmp = tmp + S.coeff[l+1]
      for j=2:n+1
        tmp = tmp + S.coeff[l+j]*x[i,j-1]
      end
      interpolates[i] = tmp
    end
    return interpolates
  end
  
  function interpolate(S::PolyharmonicSpline,x::Vector{Float64})
    return interpolate(S,x'')
  end
  
  function interpolate(S::PolyharmonicSpline,x::Vector{Float64},y::Vector{Float64})
    return interpolate(S,[x y])
  end
  
  function interpolate(S::PolyharmonicSpline,x::Vector{Float64},y::Vector{Float64},z::Vector{Float64})
    return interpolate(S,[x y z])
  end
