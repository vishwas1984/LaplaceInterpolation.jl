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
import socket

import numpy as np
from nexusformat.nexus import *
import timeit
import os.path

# this may work differently for you on the server, note that julia v 1.5.4
# should be used (v 1.6.0 is incompatible), to specify uncomment this last
# line.
from julia.api import Julia

# DIRECTORIES ###########################################################
home = str(Path.home())

# if not charlotte's home, vishwas's home
if home == '/Users/charlottehaley':
    base_dir = home + '/Documents/Data/xray/md_54_4b/'
    repo_dir = home + '/Documents/Repos/laplaceinterpolation/'
    save_data_dir = base_dir
    julia = Julia(compiled_modules=False)
elif home == '/Users/vishwasrao':
    base_dir = home + '/research/bes_project/data/'
    repo_dir = home + '/research/bes_project/repo/laplaceinterpolation/'
    save_data_dir = base_dir
    julia = Julia(compiled_modules=False)
else:
    hostname = str(socket.gethostname())
    if "nxrs" in hostname and "nxrs0" not in hostname:
        base_dir = '/data3/GUP-53547/movo2_40/md_54_4b/'
        save_data_dir = '/data3/vrao_MoVO2/'
        repo_dir = home + '/Repos/laplaceinterpolation/'
        # On nxrs, we actually use julia v 1.0 because this next line didn't work.
        # julia = Julia(runtime=home+"/julia-1.5.4/bin/julia")
        julia = Julia(compiled_modules=False)


from julia import Main
filename = base_dir + 'movo2_40_120K.nxs'
filename_background = base_dir + 'movo2_40_background.nxs'

# You need to give repo_dir in order for this to work
Main.include(repo_dir+"/GeneralMK3D.jl")

def flipaxis(a, i):
    aprime = np.swapaxes(np.swapaxes(a, 0, i)[::-1], 0, i)
    return aprime


# LOAD DATA #############################################################

data = nxload(filename)
data.unlock()

movo2_40_background = nxload(filename_background)
movo2_40_background.unlock()

hstart, hend = (-0.2, 6,2)
kstart, kend = (-0.2, 8.2)
lstart, lend = (-0.2, 8.2)

# ## get the symmetric transform data
z3d = data.entry.symm_transform[hstart:hend, kstart:kend, lstart:lend].data.nxvalue

# axes
x = data.entry.symm_transform[hstart:hend, kstart:kend, lstart:lend].Ql.nxvalue
x2 = data.entry.symm_transform[hstart:hend, kstart:kend, lstart:lend].Qk.nxvalue
x3 = data.entry.symm_transform[hstart:hend, kstart:kend, lstart:lend].Qh.nxvalue

# increment in h, k, l directions
dx = 0.02 # x[1] - x[0]
dx2 = 0.02 # x2[1] - x2[0]
dx3 = 0.02 # x3[1] - x3[0]

# ## define the symmetrizing operation and the standard punch

qh_lim = 8
qk_lim = 8
ql_lim = 6

# Number of pixels per unit cell
Qh = 50
Qk = 50
Ql = 50

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

# PUNCHING ###################################################

# To ensure efficient parallel processing for the interpolation step, the
# punching is done in Julia

radius = 0.200



def standard_punch_loop(x, x2, x3, rad):
    inds = []
    Nx = len(x)
    Ny = len(x2)
    Nz = len(x3)
    if type(rad) == float:
        rad_l = rad
        rad_k = rad
        rad_h = rad
    elif type(rad) == tuple:
        rad_l = rad[1]
        rad_k = rad[2]
        rad_h = rad[3]
    L, K, H = np.meshgrid(x, x2, x3, indexing='ij')
    for l in range(-Nx, Nx + 1):
        for k in range(-Ny, Ny + 1):
            for h in range(-Nz, Nz + 1):
                inds.append(np.argwhere((((l-L)/rad_l)**2 + ((k-K)/rad_k)**2 +
                                         ((h-H)/rad_h)**2) < 1.0))

                    

# punched = standard_punch_loop(x, x2, x3, radius)

# punched = Main.punch_holes_nexus(x, x2, x3, radius)

# # MATERN INTERPOLATION ########################################
# 
# # Set parameters
# # epsilon: regularization (smoothness) parameter
# # m: order parameter
# 
# # we will use a punch radius of 0.2
# 
# epsilon = 0.0
# m = 2
# xmin = 0
# xmax = 7
# ymin = 0
# ymax = 9
# zmin = 0
# zmax = 9
# xbegin = ybegin = zbegin = -0.2
# 
# # Create a copy in order to ensure original datasets are not overwritten
# z3d_copy = np.copy(z3d)
# z3d_restored = np.copy(z3d)
# 
# # Stride is set to 10 to ensure efficient parallel processing (see below)
# # stride = 10
# 
# # Interpolating across entire data can be time consuming. Instead, dividing
# # into small chunks results in smaller but a large number of linear systems.
# # Currently, "stride" is chosen adhoc.  For MoVO2, I chose 10 to ensure that
# # the length of the cube of the data that is sent is slightly larger than the
# # diameter of the punch. In case of Matern, different "stride" values will
# # result in different interpolation results.  Larger stride values might result
# # in "better" interpolation but this comes at a cost. For Laplace, however,
# # interpolation results are independent of the value of stride as long as
# # length of the cube of the data that is sent is slightly larger than the
# # diameter of the punch. In summary, value of "stride" depends somewhat on the
# # problem.
# starttime = timeit.default_timer()
# # restored = Main.Parallel_Matern3D_Grid(x, x2, x3, z3d_copy, epsilon, radius, dx, dx2, dx3, xmin, xmax, ymin, ymax, zmin, zmax, m)
# # restored_parallel = np.reshape(restored, (len(x3), len(x2), len(x)))
# # restored_parallel_transpose = np.transpose(restored_parallel,(2,1,0)) 
# print("Time taken for Parallel Matern interpolation with m = 2 and epsilon = 0, punch radius 0.2:", timeit.default_timer() - starttime)
# starttime = timeit.default_timer()
# 
# # for i in range(xmin, xmax):
# #     # (i2-i1)*h will be the length of the cube
# #     i1 = int((i - zbegin) / dx) - stride
# #     # Here we are sending only the cube surrounding the punch for interpolation
# #     #i2 = i1 + 2*stride + 1
# #     for j in range(ymin, ymax):
# #         j1 = int((j - ybegin)/dx2) - stride
# #         #j2 = j1 + 2*stride + 1
# #         # (j2-j1)*h will be the length of the cube. For some crystals
# #         # (j2-j1)*h != (i2-i1)*h because of different aspect ratios.
# #         for k in range(zmin, zmax):
# #             if(i==xmin or i==xmax-1 or j==ymin or j==ymax-1 or k==zmin or k==zmax-1):
# #                 stride = 10
# #             else:
# #                 stride = 20
# #             i2 = i1 + 2*stride + 1
# #             j2 = j1 + 2*stride + 1
# #             k1 = int((k - ybegin)/dx3) - stride
# #             k2 = k1 + 2*stride + 1
# #             # (k2-k1)*h will be the length of the cube.
# #             z3temp = z3d_copy[i1:i2, j1:j2, k1:k2]
# #             restored, punched = Main.Matern3D_Grid(x[i1:i2], x2[j1:j2],
# #                                                    x3[k1:k2], z3temp, epsilon,
# #                                                    radius, dx, dx2, dx3, m)
# #             restored_img_reshape = np.reshape(restored, (2*stride + 1, 2*stride + 1, 2*stride + 1))
# #             # Note the transposition is due to different ordering in Julia
# #             z3d_restored[i1:i2, j1:j2, k1:k2] = np.transpose(restored_img_reshape, (2, 1, 0))
# 
# 
# print("Time taken for Matern interpolation with m = 2 and epsilon = 0, punch radius 0.2:", timeit.default_timer() - starttime)
# 
# # The result of the Matern interpolation is in z3d_restored
# 
# # LAPLACE INTERPOLATION #################################################
# 
# # Interpolated data is in z3d_restored_laplace. Original in z3d.
# 
# # Laplace interpolation has no parameters
# 
# # Copy arrays
# z3d_copy = np.copy(z3d)
# z3d_restored_laplace = np.copy(z3d)
# 
# # Start timer
# starttime = timeit.default_timer()
# 
# # for i in range(xmin, xmax):
# #     i1 = int((i - zbegin)/dx) - stride
# #     i2 = i1 + 2*stride + 1
# #     for j in range(ymin, ymax):
# #         j1 = int((j - ybegin)/dx2) - stride
# #         j2 = j1 + 2*stride + 1
# #         for k in range(zmin, zmax):
# #             if(i==xmin or i==xmax-1 or j==ymin or j==ymax-1 or k==zmin or k==zmax-1):
# #                 stride = 10
# #             else:
# #                 stride = 20
# #             i2 = i1 + 2*stride + 1
# #             j2 = j1 + 2*stride + 1
# # 
# #             k1 = int((k - ybegin)/dx3) - stride
# #             k2 = k1 + 2*stride + 1
# #             z3temp = z3d_copy[i1:i2, j1:j2, k1:k2]
# #             restored, punched = Main.Laplace3D_Grid(x[i1:i2], x2[j1:j2],
# #                                                     x3[k1:k2], z3temp, radius,
# #                                                     dx, dx2, dx3)
# #             restored_img_reshape = np.reshape(restored, (2*stride + 1, 2*stride + 1, 2*stride + 1))
# #             # Note the transposition is due to different ordering in Julia
# #             z3d_restored_laplace[i1:i2, j1:j2, k1:k2] = np.transpose(restored_img_reshape, (2, 1, 0))
# 
# # Print time
# print("Time taken for Laplace interpolation with punch radius 0.02:", timeit.default_timer() - starttime)
# 
# 
# # PLOTTING COMMANDS (MATPLOTLIB) ###########################################
# 
# # Taking a 1D slice
# 
# # Index in x and y
# # idx = 60
# # idy = 10
# 
# # Find the maximum of the data on the slice common to both z3d_copy and z3d
# # (are these different?) and add 10 for good measure
# # max1  = np.max(z3d_copy[:,idy,idx])
# # max2  = np.max(z3d[:,idy,idx])
# # max_y = np.max([max1, max2])+10
# 
# # Plot original data, matern and laplace interpolations
# #
# # fig,ax=plt.subplots(1,3, figsize=(15,5))
# # ax[0].semilogy((z3d[:,idy, idx]))
# # ax[0].set_ylim([0.00001, max_y])
# # ax[0].set_title("Original data")
# # ax[1].semilogy((z3d_restored[:, idy,idx]))
# # ax[1].set_title("Matern interpolated")
# # ax[1].set_ylim([0.00001, max_y])
# # ax[2].semilogy((z3d_restored_laplace[:, idy,idx]))
# # ax[2].set_title("Laplace interpolated")
# # ax[2].set_ylim([0.00001, max_y])
# 
# # SAVING #################################################################
# 
# # ## Save the Matern and Laplace Interpolated data to an .nxs file in the save
# # directory
# 
# expt_data = nxload(filename)['entry']
# 
# root = NXroot(NXentry())
# stdinterp = NXfield(symmetrize(z3d_restored[0:311, 0:411, 0:411]), name='sphere_punch_matern_interp_data')
# root.entry.sphere_matern_data = NXdata(stdinterp, expt_data.symm_transform[-6.:5.98, -8.:7.98, -8.:7.98].nxaxes)
# 
# sphmat = save_data_dir + '/movo2_40_sphere_matern_data_different_stride_boundary.nxs'
# 
# if os.path.exists(sphmat):
#     os.remove(sphmat)
# 
# 
# root.save(sphmat)
# 
# root = NXroot(NXentry())
# stdinterp = NXfield(symmetrize(z3d_restored_laplace[0:311, 0:411, 0:411]), name='sphere_punch_laplace_interp_data')
# root.entry.sphere_laplace_data = NXdata(stdinterp, expt_data.symm_transform[-6.:5.98, -8.:7.98, -8.:7.98].nxaxes)
# 
# sphlap = save_data_dir + '/movo2_40_sphere_laplace_data_different_stride_boundary.nxs'
# 
# if os.path.exists(sphlap):
#     os.remove(sphlap)
# 
# 
# root.save(sphlap)
# 
# 
# stdinterp = NXfield(symmetrize(restored_parallel_transpose[0:311, 0:411, 0:411]), name='sphere_punch_parallelmatern_interp_data')
# root.entry.sphere_parallelmatern_data = NXdata(stdinterp, expt_data.symm_transform[-6.:5.98, -8.:7.98, -8.:7.98].nxaxes)
# 
# sphparmat = save_data_dir + '/movo2_40_sphere_parallelmatern_interp.nxs'
# 
# if os.path.exists(sphparmat):
#     os.remove(sphparmat)
# 
# 
# root.save(sphparmat)
# 
# print("Files saved in: ", save_data_dir)
# 
# # EOF ########################################################################
