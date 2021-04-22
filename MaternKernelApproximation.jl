# This code interpolates for the missing points in an image. The code is specifically designed for removing Bragg peaks using the punch and fill algorithm. This code needs the image and the coordinates where the Bragg peaks needs to be removed and the radius (which can be the approximate width of the peaks). The code assumes all the "punches" will be of the same size and there are no Bragg peaks on the boundaries. Lines 2 to ~ 175 consists of helper functions and 175 onwards corresponds to the driver code.
using LinearAlgebra, SparseArrays
using TestImages, Colors, Plots, FileIO, JLD, BenchmarkTools
#include("PyAMG.jl")

function spdiagm_nonsquare(m, n, args...)
    I, J, V = SparseArrays.spdiagm_internal(args...)
    return sparse(I, J, V, m, n)
end

#Constructing the 2D Laplace matrix
# returns -∇² (discrete Laplacian, real-symmetric positive-definite) on n₁×n₂ grid
function ∇²(n₁,n₂)
    o₁ = ones(n₁)
    ∂₁ = spdiagm_nonsquare(n₁+1,n₁,-1=>-o₁,0=>o₁)
    o₂ = ones(n₂)
    ∂₂ = spdiagm_nonsquare(n₂+1,n₂,-1=>-o₂,0=>o₂)
    return kron(sparse(I,n₂,n₂), ∂₁'*∂₁) + kron(∂₂'*∂₂, sparse(I,n₁,n₁))
end

#Constructing the 3D Laplace matrix
# function spdiagm_nonsquare(m, n, args...)
#     I, J, V = SparseArrays.spdiagm_internal(args...)
#     return sparse(I, J, V, m, n)
# end

# returns -∇² (discrete Laplacian, real-symmetric positive-definite) on n₁×n₂ grid
function ∇²3d(n₁,n₂,n3)
    o₁ = ones(n₁)
    ∂₁ = spdiagm_nonsquare(n₁+1,n₁,-1=>-o₁,0=>o₁)
    o₂ = ones(n₂)
    ∂₂ = spdiagm_nonsquare(n₂+1,n₂,-1=>-o₂,0=>o₂)
    O3 = ones(n3)
    del3 = spdiagm_nonsquare(n3+1,n3,-1=>-O3,0=>O3)
    return kron(sparse(I,n3,n3),sparse(I,n₂,n₂), ∂₁'*∂₁) + kron(sparse(I,n3,n3), ∂₂'*∂₂, sparse(I,n₁,n₁)) + kron(del3'*del3, sparse(I,n₂,n₂), sparse(I,n₁,n₁))
end

function ∇²3d_Grid(n₁,n₂,n3, h)
    o₁ = ones(n₁)/h
    ∂₁ = spdiagm_nonsquare(n₁+1,n₁,-1=>-o₁,0=>o₁)
    o₂ = ones(n₂)/h
    ∂₂ = spdiagm_nonsquare(n₂+1,n₂,-1=>-o₂,0=>o₂)
    O3 = ones(n3)/h
    del3 = spdiagm_nonsquare(n3+1,n3,-1=>-O3,0=>O3)
    # sx_sparse = sparse(I,n₁,n₁);
    # sy_sparse = sparse(I,n₂,n₂);
    # sz_sparse = sparse(I,n3,n3);
    #A1 = 1/h^2*(kron(sparse(I,n3,n3),sparse(I,n₂,n₂), ∂₁'*∂₁));


    return (kron(sparse(I,n3,n3),sparse(I,n₂,n₂), ∂₁'*∂₁) + kron(sparse(I,n3,n3), ∂₂'*∂₂, sparse(I,n₁,n₁)) + kron(del3'*del3, sparse(I,n₂,n₂), sparse(I,n₁,n₁)))
end

function return_boundary_nodes(xpoints, ypoints, zpoints)
    BoundaryNodes3D =[];
    counter = 0;
    for k = 1:zpoints
        for j = 1:ypoints
            for i = 1:xpoints
                counter=counter+1;
                if(k == 1 || k == zpoints || j == 1|| j == ypoints || i == 1 || i == xpoints)
                    BoundaryNodes3D = push!(BoundaryNodes3D, counter)
                end
            end
        end
    end
    return BoundaryNodes3D
end

function return_boundary_nodes2D(xpoints, ypoints)
    BoundaryNodes2D =[];
    counter = 0;
    
    for j = 1:ypoints
        for i = 1:xpoints
            counter=counter+1;
            if( j == 1|| j == ypoints || i == 1 || i == xpoints)
                BoundaryNodes2D = push!(BoundaryNodes2D, counter)
            end
        end
    end
    
    return BoundaryNodes2D
end

# NB: Have to make sure the punch is not punching on the boundary 
# Make sure not to pass in data such that there will be missing data on the boundary
# This is a todo still.
# THis note refers to the next two funcitons punch_holes

function punch_holes_nexus(xpoints, ypoints, zpoints, radius)
    xlen = length(xpoints);
    ylen = length(ypoints);
    zlen = length(zpoints);
    masking_data_points = [];
    absolute_indices = Int64[];

    #Making sure that boundary is not punched
    xmax = maximum(xpoints);
    ymax = maximum(ypoints);
    zmax = maximum(zpoints);
    xmin = minimum(xpoints);
    ymin = minimum(ypoints);
    zmin = minimum(zpoints);


    count = 1;
    for i = 1:zlen
        ir = round(zpoints[i]);
        for j = 1:ylen
            jr = round(ypoints[j]);
            for h = 1:xlen
                hr = round(xpoints[h]);
                if(ir>=zmin+1 && ir <= zmax-1 && jr >= ymin+1 && jr <= ymax-1 && hr >= xmin+1 && hr <= xmax+1)
                    if((hr-xpoints[h])^2 + (jr-ypoints[j])^2/(1.5^2) + (ir - zpoints[i])^2/(1.5^2) <= radius^2)
                        #imgg_copy[h,j,i] = 1
                        #append!(masking_data_points,[(h,j,i)]);
                        append!(absolute_indices, count);
                            
                    end
                end
                count = count +1;
            end
        end
    end

    return absolute_indices

end


function punch_holes_3D(centers, radius, xpoints, ypoints, zpoints)
    clen = length(centers);
    masking_data_points = [];
    absolute_indices = Int64[];

    for a = 1:clen
        c=centers[a];
        count = 1;
        for i = 1:zpoints
            for j = 1:ypoints
                for h = 1:xpoints
                    if((h-c[1])^2 + (j-c[2])^2 + (i - c[3])^2 <= radius^2)
                        #imgg_copy[h,j,i] = 1
                        append!(masking_data_points,[(h,j,i)]);
                        append!(absolute_indices, count);
                            
                    end
                    count = count +1;
                end
            end
        end
    end
    return absolute_indices

end

function punch_holes_2D(centers, radius, xpoints, ypoints)
    clen = length(centers);
    masking_data_points = [];
    absolute_indices = Int64[];
    

    for a = 1:clen
        c=centers[a];
       
        count = 1;
        
        for j = 1:ypoints
            for h = 1:xpoints
                if((h-c[1])^2 + (j-c[2])^2  <= radius^2)
                    #imgg_copy[h,j,i] = 1
                    append!(masking_data_points,[(h,j)]);
                    append!(absolute_indices, count);
                        
                end
                count = count +1;
            end
        end
        
    end
    return absolute_indices

end

function Matern1D(h,N,f_array, args...)
    A= Tridiagonal([fill(-1.0/h^2, N-2); 0], [1.0; fill(2.0/h^2, N-2); 1.0], [0.0; fill(-1.0/h^2, N-2);]);
    sizeA = size(A,1);
    epsilon = 2.2
    for i = 1:sizeA
        A[i,i] = A[i,i] + epsilon^2
    end
    A2 = A*A;
    diag_c = ones(N);
    for i in discard
        diag_c[i] = 0;
    end
    #diag_c[discard] .= 0;
    C = diagm(diag_c);
    Id = Matrix(1.0I, N,N);
    return (C-(Id -C)*A2)\(C*f_array);
end

function Matern2D(xpoints, ypoints, imgg, epsilon, centers, radius, args...)
    A2D = ∇²(xpoints, ypoints);

    BoundaryNodes = return_boundary_nodes2D(xpoints, ypoints);
    for i in BoundaryNodes
        rowindices = A2D.rowval[nzrange(A2D, i)];
        A2D[rowindices,i].=0;
        A2D[i,i] = 1.0;
    end

    sizeA = size(A2D,1);
    for i = 1:sizeA
        A2D[i,i] = A2D[i,i] + epsilon^2
    end
    A2DMatern = A2D*A2D;

    discard = punch_holes_2D(centers, radius, xpoints, ypoints);

    punched_image = copy(imgg);
    punched_image[discard] .= 1;

    totalsize = prod(size(imgg));
    C = sparse(I, totalsize, totalsize)
    for i in discard
        C[i,i] = 0;
    end
    #C[discard,discard] .= 0
    Id = sparse(I, totalsize, totalsize);
    f = punched_image[:];
    C*f
    #u =((C-(Id -C)*A3DGiphy)) \ (C*f);
    #restored_img = reshape(u, xpoints, ypoints, zpoints);

    rhs_a = C*f;

    rhs_a = Float64.(rhs_a);

    u =((C-(Id -C)*A2DMatern)) \ rhs_a;
    #Amat = ((C-(Id -C)*A2DMatern));
    #u = PyAMG.solve(Amat, rhs_a);

    restored_img = reshape(u, xpoints, ypoints);
    restored_img = Gray.(restored_img);
    return restored_img, punched_image;
end

function Matern3D(xpoints, ypoints, zpoints, imgg, epsilon, centers, radius, args...)
    A3D = ∇²3d(xpoints, ypoints, zpoints);

    BoundaryNodes = return_boundary_nodes(xpoints, ypoints, zpoints);
    for i in BoundaryNodes
        rowindices = A3D.rowval[nzrange(A3D, i)];
        A3D[rowindices,i].=0;
        A3D[i,i] = 1.0;
    end

    sizeA = size(A3D,1);
    for i = 1:sizeA
        A3D[i,i] = A3D[i,i] + epsilon^2
    end
    A3DMatern = A3D*A3D;

    discard = punch_holes_3D(centers, radius, xpoints, ypoints, zpoints);

    punched_image = copy(imgg);
    punched_image[discard] .= 1;

    totalsize = prod(size(imgg));
    C = sparse(I, totalsize, totalsize)
    for i in discard
        C[i,i] = 0;
    end
    #C[discard,discard] .= 0
    Id = sparse(I, totalsize, totalsize);
    f = punched_image[:];
    C*f
    #u =((C-(Id -C)*A3DGiphy)) \ (C*f);
    #restored_img = reshape(u, xpoints, ypoints, zpoints);

    rhs_a = C*f;

    rhs_a = Float64.(rhs_a);

    u =((C-(Id -C)*A3DMatern)) \ rhs_a;

    restored_img = reshape(u, xpoints, ypoints, zpoints);
    restored_img = Gray.(restored_img);
    return restored_img, punched_image;
end

function Matern3D_Grid(xpoints, ypoints, zpoints, imgg, epsilon, radius, h, args...)
    xlen = length(xpoints);
    ylen = length(ypoints);
    zlen = length(zpoints);
    A3D = ∇²3d_Grid(xlen, ylen, zlen, h);

    BoundaryNodes = return_boundary_nodes(xlen, ylen, zlen);
    for i in BoundaryNodes
        rowindices = A3D.rowval[nzrange(A3D, i)];
        A3D[rowindices,i].=0;
        A3D[i,i] = 1.0;
    end

    sizeA = size(A3D,1);
    for i = 1:sizeA
        A3D[i,i] = A3D[i,i] + epsilon^2
    end
    A3DMatern = A3D*A3D;

    discard = punch_holes_nexus(xpoints, ypoints, zpoints, radius);

    punched_image = copy(imgg);
    punched_image[discard] .= 1;

    totalsize = prod(size(imgg));
    C = sparse(I, totalsize, totalsize)
    rhs_a = punched_image[:];
    for i in discard
        C[i,i] = 0;
        rhs_a[i] = 0;
    end
    #C[discard,discard] .= 0
    Id = sparse(I, totalsize, totalsize);
    
    #u =((C-(Id -C)*A3DGiphy)) \ (C*f);
    #restored_img = reshape(u, xpoints, ypoints, zpoints);

    #rhs_a = C*f;

    #rhs_a = Float64.(rhs_a);
    # Amat = ((C-(Id -C)*A3DMatern));
    # u = PyAMG.solve(Amat, rhs_a);

u =((C-(Id -C)*A3DMatern)) \ rhs_a;

    #restored_img = reshape(u, xlen, ylen, zlen);
    # restored_img = Gray.(restored_img);
    return u, punched_image[:];
end

function Laplace3D_Grid(xpoints, ypoints, zpoints, imgg, epsilon, radius, h, args...)
    xlen = length(xpoints);
    ylen = length(ypoints);
    zlen = length(zpoints);
    A3D = ∇²3d_Grid(xlen, ylen, zlen, h);
    

    BoundaryNodes = return_boundary_nodes(xlen, ylen, zlen);
    for i in BoundaryNodes
        rowindices = A3D.rowval[nzrange(A3D, i)];
        A3D[rowindices,i].=0;
        A3D[i,i] = 1.0;
    end

    discard = punch_holes_nexus(xpoints, ypoints, zpoints, radius);

    punched_image = copy(imgg);
    punched_image[discard] .= 1;

    totalsize = prod(size(imgg));
    C = sparse(I, totalsize, totalsize)
    rhs_a = punched_image[:];
    for i in discard
        C[i,i] = 0;
        rhs_a[i] = 0;
    end
    #C[discard,discard] .= 0
    Id = sparse(I, totalsize, totalsize);
    
    #u =((C-(Id -C)*A3DGiphy)) \ (C*f);
    #restored_img = reshape(u, xpoints, ypoints, zpoints);

    #rhs_a = C*f;

    #rhs_a = Float64.(rhs_a);
    # Amat = (Id -C)*A3D -C;
    # u = PyAMG.solve(Amat, -rhs_a);
    u =((C-(Id -C)*A3D)) \ rhs_a;

    #return u, punched_image[:], A3D;
    return u, punched_image[:];
end


#2D Example: Mandrill

# img = testimage("mandrill");

# imgg = Gray.(img);

# mat = convert(Array{Float64}, imgg)[1:256,1:512];
# # This image is square
# plot(imgg)
# cent = [(100, 200), (200, 100), (200, 400)]
# radius = 20;
# xpoints = size(mat,1);
# ypoints = size(mat,2);
# epsilon = 0.2
# restored_image, punched_image =  Matern2D(xpoints, ypoints, mat, epsilon, cent, radius);

# # BenchmarkTools.DEFAULT_PARAMETERS.seconds = 100;
# # @benchmark Matern2D(xpoints, ypoints, mat, epsilon, cent, radius);

# plot(Gray.(restored_image), title="restored Image")

# obj = load("/Users/vishwasrao/Research/BES_Project/Repo/laplaceinterpolation/cat_bow.gif")
# obj_copy = load("/Users/vishwasrao/Research/BES_Project/Repo/laplaceinterpolation/cat_bow.gif")

# imgg = Gray.(obj);
# imgg_copy = Gray.(obj_copy);

# cent = [(50, 50,30), (50, 100,30), (50, 150,30), (100, 50, 30),
#     (100, 100,30), (100, 150, 30)];

# radius = 20;
# epsilon = 0.1;
# xpoints = size(imgg, 1);
# ypoints = size(imgg, 2);
# zpoints = size(imgg, 3);

# # @benchmark restored_image, punched_image  = Matern3D(xpoints, ypoints, zpoints, imgg, epsilon, cent, radius);
# print()
# BenchmarkTools.DEFAULT_PARAMETERS.seconds = 600;
# BenchmarkTools.DEFAULT_PARAMETERS.samples = 50;
# @benchmark Matern3D(xpoints, ypoints, zpoints, imgg, epsilon, cent, radius);

# restored_image = Gray.(restored_image);
# plot1 = plot(imgg[:,:,15], title = "Original Image");
# plot2 = plot(restored_image[:,:,15], title = "Restored Image");
# plot3 = plot(punched_image[:,:,15], title = "Punched Image");
# plot(plot1, plot3, plot2, layout = (1, 3), legend = false)







