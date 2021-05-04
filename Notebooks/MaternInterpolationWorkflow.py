#!/usr/bin/env python
# coding: utf-8


# PREAMBLE ###############################################################

# Demo of Laplacian and Matern interpolation on Vanadium Dioxide Data
#
# We will be comparing the result of standard punch and fill to that of the
# technique involving watershed segmentation of the Bragg peaks followed by
# Laplacian and Matern fill.
#
# the data can be found on nxrs:
# `/data3/GUP-53547/movo2_40/md_54_4b/movo2_40_120k.nxs`
#
# the most relevant scattering is found half-integer l planes, with
# only weak size-effect scattering close to the bragg peaks; a 3d-deltapdf
# should show a clear 2d 'x' pattern.

# LOAD LIBRARIES #######################################################

from pathlib import Path
# import socket

import numpy as np
from nexusformat.nexus import *
import timeit

# this may work differently for you on the server, note that julia v 1.5.4
# should be used (v 1.6.0 is incompatible), to specify uncomment this last
# line.
from julia.api import Julia
julia = Julia(compiled_modules=False)
from julia import Main
# julia = Julia(runtime="/home/chaley/julia-1.5.4/bin/julia")

# DIRECTORIES ###########################################################
home = str(Path.home())
# hostname = str(socket.gethostname())

# if not charlotte's home, vishwas's home
if home == '/Users/charlottehaley':
    base_dir = home + '/documents/data/xray/md_54_4b/'
    repo_dir = home + '/documents/repos/laplaceinterpolation/'
    save_data_dir = base_dir
elif home == '/Users/vishwasrao':
    base_dir = home + '/research/bes_project/data/'
    repo_dir = home + '/research/bes_project/repo/laplaceinterpolation/'
    save_data_dir = base_dir
else:
    # if "nxrs" in hostname and "nxrs0" not in hostname:
    base_dir = '/data3/GUP-53547/movo2_40/md_54_4b/'
    save_data_dir = home


filename = base_dir + 'movo2_40_120k.nxs'
filename_background = base_dir + 'movo2_40_background.nxs'

# You need to give repo_dir in order for this to work
Main.include(repo_dir+"/MaternKernelApproximation.jl")


def flipaxis(a, i):
    aprime = np.swapaxes(np.swapaxes(a, 0, i)[::-1], 0, i)
    return aprime


# LOAD DATA #############################################################

data = nxload(filename)
data.unlock()

movo2_40_background = nxload(filename_background)
movo2_40_background.unlock()

# ## get the symmetric transform data
z3d = data.entry.symm_transform[-0.2:6.2, -0.2:8.2, -0.2:8.2].data.nxvalue

# axes
x = data.entry.symm_transform[-0.2:6.2, -0.2:8.2, -0.2:8.2].Ql.nxvalue
x2 = data.entry.symm_transform[-0.2:6., -0.2:8.2, -0.2:8.2].Qk.nxvalue
x3 = data.entry.symm_transform[-0.2:6., -0.2:8.2, -0.2:8.2].Qh.nxvalue

# increment in h, k, l directions
dx = 0.02 # x[1] - x[0]
dx2 = 0.02 # x2[1] - x2[0]
dx3 = 0.02 # x3[1] - x3[0]


# ## define the symmetrizing operation and the standard punch

qh_lim = 8
qk_lim = 8
ql_lim = 6

kmin = 50*(6-ql_lim)
kmax = 50*(6+ql_lim)
jmin = 50*(8-qk_lim)
jmax = 50*(8+qk_lim)
imin = 50*(8-qh_lim)
imax = 50*(8+qh_lim)


def symmetrize(res):
    vvals = np.zeros((601, 801, 801))
    vvals[290:601, 390:801, 390:801] = res
    vvals[300:601, 400:801, 0:401] = flipaxis(vvals[300:601, 400:801, 400:801],
                                              2)
    vvals[300:601, 0:401, 0:801] = flipaxis(vvals[300:601, 400:801, 0:801], 1)
    vvals[0:301, 0:801, 0:801] = flipaxis(vvals[300:601, 0:801, 0:801], 0)
    return vvals[0:600, 0:800, 0:800]


# MATERN INTERPOLATION ########################################

# Set parameters
# epsilon: regularization (smoothness) parameter
# m: order parameter

# we will use a punch radius of 0.2

epsilon = 0.
m = 2
radius = 0.200

xmin = 0
xmax = 7
ymin = 0
ymax = 9
zmin = 0
zmax = 9
xbegin = ybegin = zbegin = -0.2

# Create a copy in order to ensure original datasets are not overwritten
z3d_copy = np.copy(z3d)
z3d_restored = np.copy(z3d)

# Stride is set to 10 to ensure efficient parallel processing (see below)
stride = 10

# Interpolating across entire data can be time consuming. Instead, dividing
# into small chunks results in smaller but a large number of linear systems.
# Currently, "stride" is chosen adhoc.  For MoVO2, I chose 10 to ensure that
# the length of the cube of the data that is sent is slightly larger than the
# diameter of the punch. In case of Matern, different "stride" values will
# result in different interpolation results.  Larger stride values might result
# in "better" interpolation but this comes at a cost. For Laplace, however,
# interpolation results are independent of the value of stride as long as
# length of the cube of the data that is sent is slightly larger than the
# diameter of the punch. In summary, value of "stride" depends somewhat on the
# problem.

starttime = timeit.default_timer()

for i in range(zmin, zmax):
    # (i2-i1)*h will be the length of the cube
    i1 = int((i - zbegin) / dx) - stride
    # Here we are sending only the cube surrounding the punch for interpolation
    i2 = i1 + 2*stride + 1
    for j in range(ymin, ymax):
        j1 = int((j - ybegin)/dx2) - stride
        j2 = j1 + 2*stride + 1
        # (j2-j1)*h will be the length of the cube. For some crystals
        # (j2-j1)*h != (i2-i1)*h because of different aspect ratios.
        for k in range(xmin, xmax):
            k1 = int((k - ybegin)/dx3) - stride
            k2 = k1 + 2*stride + 1
            # (k2-k1)*h will be the length of the cube.
            z3temp = z3d_copy[k1:k2, j1:j2, i1:i2]
            restored, punched = Main.Matern3D_Grid(x[k1:k2], x2[j1:j2],
                                                   x3[i1:i2], z3temp, epsilon,
                                                   radius, dx, dx2, dx3, m)
            restored_img_reshape = np.reshape(restored, (2*stride + 1, 2*stride + 1, 2*stride + 1))
            # Note the transposition is due to different ordering in Julia
            z3d_restored[k1:k2, j1:j2, i1:i2] = restored_img_reshape.T


print("Time taken for Matern interpolation with m = 2 and epsilon = 0, punch radius 0.02:", timeit.default_timer() - starttime)

# The result of the Matern interpolation is in z3d_restored

# LAPLACE INTERPOLATION #################################################

# Interpolated data is in z3d_restored_laplace. Original in z3d.

# Laplace interpolation has no parameters

# Copy arrays
z3d_copy = np.copy(z3d)
z3d_restored_laplace = np.copy(z3d)

# Start timer
starttime = timeit.default_timer()

for i in range(zmin, zmax):
    i1 = int((i - zbegin)/dx) - stride
    i2 = i1 + 2*stride + 1
    for j in range(ymin, ymax):
        j1 = int((j - ybegin)/dx2) - stride
        j2 = j1 + 2*stride + 1
        for k in range(xmin, xmax):
            k1 = int((k - ybegin)/dx3) - stride
            k2 = k1 + 2*stride + 1
            z3temp = z3d_copy[k1:k2, j1:j2, i1:i2]
            restored, punched = Main.Laplace3D_Grid(x[k1:k2], x2[j1:j2],
                                                    x3[i1:i2], z3temp, radius,
                                                    dx, dx2, dx3)
            restored_img_reshape = np.reshape(restored, (2*stride + 1, 2*stride + 1, 2*stride + 1))
            z3d_restored_laplace[k1:k2, j1:j2, i1:i2] = restored_img_reshape.T

# Print time
print("Time taken for Laplace interpolation with punch radius 0.02:", timeit.default_timer() - starttime)


# PLOTTING COMMANDS (MATPLOTLIB) ###########################################

# Taking a 1D slice

# Index in x and y
# idx = 60
# idy = 10

# Find the maximum of the data on the slice common to both z3d_copy and z3d
# (are these different?) and add 10 for good measure
# max1  = np.max(z3d_copy[:,idy,idx])
# max2  = np.max(z3d[:,idy,idx])
# max_y = np.max([max1, max2])+10

# Plot original data, matern and laplace interpolations
#
# fig,ax=plt.subplots(1,3, figsize=(15,5))
# ax[0].semilogy((z3d[:,idy, idx]))
# ax[0].set_ylim([0.00001, max_y])
# ax[0].set_title("Original data")
# ax[1].semilogy((z3d_restored[:, idy,idx]))
# ax[1].set_title("Matern interpolated")
# ax[1].set_ylim([0.00001, max_y])
# ax[2].semilogy((z3d_restored_laplace[:, idy,idx]))
# ax[2].set_title("Laplace interpolated")
# ax[2].set_ylim([0.00001, max_y])

# SAVING #################################################################

# ## Save the Matern and Laplace Interpolated data to an .nxs file in the save
# directory

expt_data = nxload(save_data_dir + 'movo2_40_120K.nxs')['entry']

root = NXroot(NXentry())
stdinterp = NXfield(symmetrize(z3d_restored[0:311, 0:411, 0:411]), name='sphere_punch_matern_interp_data')
root.entry.sphere_matern_data = NXdata(stdinterp, expt_data.symm_transform[-6.:5.98, -8.:7.98, -8.:7.98].nxaxes)

root.save(save_data_dir + 'movo2_40_sphere_matern_data.nxs')

root = NXroot(NXentry())
stdinterp = NXfield(symmetrize(z3d_restored_laplace[0:311, 0:411, 0:411]), name='sphere_punch_laplace_interp_data')
root.entry.sphere_laplace_data = NXdata(stdinterp, expt_data.symm_transform[-6.:5.98, -8.:7.98, -8.:7.98].nxaxes)

root.save(save_data_dir + 'movo2_40_sphere_laplace_data.nxs')

print("Files saved in: ", save_data_dir)

# EOF ########################################################################
