# This code interpolates for the missing points in an image. The code is specifically designed for removing Bragg peaks using the punch and fill algorithm. This code needs the image and the coordinates where the Bragg peaks needs to be removed and the radius (which can be the approximate width of the peaks). The code assumes all the "punches" will be of the same size and there are no Bragg peaks on the boundaries. Lines 2 to ~ 175 consists of helper functions and 175 onwards corresponds to the driver code.
using Laplacians, LinearAlgebra, SparseArrays
using TestImages, Colors, Plots, FileIO, JLD, BenchmarkTools

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

function punch_holes_3D(centers, radius, xpoints, ypoints, zpoints)
    clen = length(centers);
    masking_data_points = [];
    absolute_indices = Int64[];

    for a = 1:clen
        c=cent[a];
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

function Matern2D(rows, columns, discard, mat, BoundaryNodes, args...)
    A = ∇²(rows,columns);
    epsilon = 0.3;
    sizeA = size(A,1);
    for i = 1:sizeA
        A[i,i] = A[i,i] + epsilon^2
    end
    C = sparse(I, rows*columns, rows*columns)
    # A[BoundaryNodes,:] .= 0
    # A[:,BoundaryNodes] .= 0
    
    A[BoundaryNodes, BoundaryNodes] = sparse(I, length(BoundaryNodes), length(BoundaryNodes));
    for i in BoundaryNodes
        rowindices = A.rowval[nzrange(A, i)];
        A[rowindices,i].=0;
        A[i,i] = 1.0
    end
    
    
    for i in discard
        C[i,i] =0.
    end
    A2 = A*A;
    #C[discard,discard] .= 0
    Id = sparse(I, rows*columns,rows*columns);
    f = mat[:];
    return ((C-(Id -C)*A2)) \ (C*f);
    # ml = ruge_stuben(((C-(Id -C)*A2)))
    # u_amg = solve(ml, (C*f))
    #return restored_img = reshape(u, size(mat,1), size(mat,2));
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


obj = load("/Users/vishwasrao/Research/BES_Project/Repo/laplaceinterpolation/cat_bow.gif")
obj_copy = load("/Users/vishwasrao/Research/BES_Project/Repo/laplaceinterpolation/cat_bow.gif")

imgg = Gray.(obj);
imgg_copy = Gray.(obj_copy);

cent = [(50, 50,30), (50, 100,30), (50, 150,30), (100, 50, 30),
    (100, 100,30), (100, 150, 30)];

radius = 20;
epsilon = 0.1;
xpoints = size(imgg, 1);
ypoints = size(imgg, 2);
zpoints = size(imgg, 3);

restored_image, punched_image  = Matern3D(xpoints, ypoints, zpoints, imgg, epsilon, cent, radius);

restored_image = Gray.(restored_image);
plot1 = plot(imgg[:,:,15], title = "Original Image");
plot2 = plot(restored_image[:,:,15], title = "Restored Image");
plot3 = plot(punched_image[:,:,15], title = "Punched Image");
plot(plot1, plot3, plot2, layout = (1, 3), legend = false)







