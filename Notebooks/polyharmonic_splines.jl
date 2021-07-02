
function interpolate(S::PolyharmonicSpline,x::Array{Float64,2})
    m,n = size(x)
  
    n != S.dim && throw(DimensionMismatch("$m != $(S.dim)"))
  
    interpolates = zeros(m)
    for i=1:m
      tmp = 0.0
      l = length(S.coeff)-(n+1)
      for j=1:l
        tmp = tmp + S.coeff[j]*PolyharmonicSplines.polyharmonicK(norm(x[i,:] .- S.centers[j,:]),S.order)
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
