# Note: here we are not calling from python

# All of this is hard-coded. This needs to be changed.
xmin = 1.0
xmax = 6.0
ymin = 1.0
ymax = 8.0
zmin = 1.0
zmax = 8.0
epsilon = 0.0
radius = 0.2
xbegin = ybegin = zbegin = -0.2
z3d_copy = copy(z3d)
z3d_restored = copy(z3d)
stride = 20
h = 0.02

# Set up a list of indices that describe the square boxes
cartesian_product_boxes = []
for i = zmin:zmax - 1
    i1 = Int((i - zbegin) / h) - stride
    i2 = i1 + 2 * stride
    for j = ymin:ymax - 1
        j1 = Int((j - ybegin) / h) - stride
        j2 = j1 + 2 * stride
        for k = xmin:xmax - 1
            k1 = Int((k - ybegin) / h) - stride
            k2 = k1 + 2 * stride
            append!(cartesian_product_boxes, [(i1, i2, j1, j2, k1, k2)])
        end
    end
end

Threads.@threads for i = 1:length(cartesian_product_boxes)
#=    
    i1 = cartesian_product_boxes[i][1]
    i2 = cartesian_product_boxes[i][2]
    j1 = cartesian_product_boxes[i][3]
    j2 = cartesian_product_boxes[i][4]
    k1 = cartesian_product_boxes[i][5]
    k2 = cartesian_product_boxes[i][6]
=#
    (i1, i2, j1, j2, k1, k2) = cartesian_product_boxes[i]
    z3temp = z3d_copy[k1 + 1:k2, j1 + 1:j2, i1 + 1:i2]
    restored_img, punched_image = Laplace3D_Grid(x[k1 + 1:k2], x2[j1 + 1:j2], x3[i1 + 1:i2], z3temp, epsilon, radius, h)
    z3d_restored[k1 + 1:k2, j1 + 1:j2, i1 + 1:i2] = reshape(restored_img, (2 * stride, 2 * stride, 2 * stride))
end
