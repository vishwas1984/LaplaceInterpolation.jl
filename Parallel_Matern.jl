using NPZ
include("MaternKernelApproximation.jl")
z3d = npzread("Notebooks/volume_data_movo.npy");
x = npzread("Notebooks/xaxis.npy");
x2 = npzread("Notebooks/yaxis.npy");
x3 = npzread("Notebooks/zaxis.npy");
xmin = 1;
xmax = 6;
ymin = 1;
ymax = 8;
zmin = 1;
zmax = 8;
epsilon = 0;
radius = 0.2;
xbegin = ybegin = zbegin =-0.2;
z3d_copy = copy(z3d);
z3d_restored = copy(z3d);
stride = 20;
h = 0.02;
# starttime = timeit.default_timer()
Threads.@threads for i = zmin:zmax-1
    i1 = Int((i-zbegin) /h)-stride;
    i2 = i1+2*stride;
    #print(i1,i2)
    for j = ymin:ymax-1
        j1 = Int((j-ybegin)/h)-stride;
        j2 = j1+2*stride;
        #print(j1,j2)
        for k = xmin:xmax-1
            k1 = Int((k-ybegin)/h) - stride;
            k2 = k1+2*stride;
            #print(k1,k2)
            z3temp = z3d_copy[k1+1:k2,j1+1:j2,i1+1:i2];
            restored_img, punched_image = Laplace3D_Grid(x[k1+1:k2], x2[j1+1:j2], x3[i1+1:i2], z3temp, epsilon, radius, h);
            restored_img_reshape = reshape(restored_img, (2*stride,2*stride,2*stride));
            z3d_restored[k1+1:k2, j1+1:j2, i1+1:i2] = restored_img_reshape;
        end
    end
end


# no_of_threads = [1, 2, 4, 10, 20, 40]
# times= [55, 29.61,20.0, 12.60, 12.623, 13.651]
# perfect_scaling = [55, 27.5, 13.75, 5.5, 2.75, 1.375]
# plt.loglog(no_of_threads, times, '--o')
# plt.loglog(no_of_threads, perfect_scaling, '-.v')
# plt.xlabel('No of Threads')
# plt.ylabel('Times')
# plt.legend(['Laplace timings', 'Perfect Scaling'])