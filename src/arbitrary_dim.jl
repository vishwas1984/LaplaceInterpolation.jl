
""" Give the boundary nodes """
function bdy_nodes(dims::Tuple{Int64})
    bdy = []
    D = length(dims)
    for (i, d) in enumerate(dims)
        push!(bdy, CartesianIndices((dims[1:(i-1)]..., 1, dims[(i+1):D]...))[:]...)
        push!(bdy, CartesianIndices((dims[1:(i-1)]..., d, dims[(i+1):D]...))[:]...)
    end
    return bdy
end

# function nablasq_arb(dims)
    
