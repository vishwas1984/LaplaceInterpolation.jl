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
from   astropy.convolution import convolve, Kernel, Gaussian1DKernel

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
    base_dir = home + '/Documents/Data/Xray/md_54_4b/'
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

# ## get the symmetric transform data
z3d = data.entry.symm_transform[-0.5:6.498, -0.5:8.498, -0.5:8.498].data.nxvalue

# axes
x = data.entry.symm_transform[-0.5:6.498, -0.5:8.498, -0.5:8.498].Ql.nxvalue
x2 = data.entry.symm_transform[-0.5:6.498, -0.5:8.498, -0.5:8.498].Qk.nxvalue
x3 = data.entry.symm_transform[-0.5:6.498, -0.5:8.498, -0.5:8.498].Qh.nxvalue

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
    vvals=np.zeros((601,801,801))
    vvals[275:601,375:801,375:801] = res
    vvals[300:601,400:801,0:401] = flipaxis(vvals[300:601,400:801,400:801],2)
    vvals[300:601,0:401,0:801] = flipaxis(vvals[300:601,400:801,0:801],1)
    vvals[0:301,0:801,0:801] = flipaxis(vvals[300:601,0:801,0:801],0)
    return vvals

# PUNCHING ###################################################

# To ensure efficient parallel processing for the interpolation step, the
# punching is done in Julia

radius = 0.2001

radius_h = radius
radius_k = radius
radius_l = radius

# punch_locs = standard_punch(x, x2, x3, (radius_h, radius_k, radius_l))

# punched = punch_locs*z3d

def standard_lu(x, x2, x3, rad):
    """ Get the punch on the standard lattice unit """
    L, K, H = np.meshgrid(x, x2, x3, indexing='ij')
    punch_locs = np.where((((L - np.rint(L)) ** 2 / (rad[0] ** 2) +
                             (K - np.rint(K)) ** 2 / (rad[1] ** 2) +
                             (H - np.rint(H)) ** 2 / (rad[2] ** 2)) < 1.0),
                            np.ones((len(x), len(x2), len(x3))),
                            np.zeros((len(x), len(x2), len(x3))))
    return punch_locs


standard_x = data.entry.symm_transform[-0.5:0.5, -0.5:0.5, -0.5:0.5].Ql.nxvalue
standard_y = data.entry.symm_transform[-0.5:0.5, -0.5:0.5, -0.5:0.5].Qk.nxvalue
standard_z = data.entry.symm_transform[-0.5:0.5, -0.5:0.5, -0.5:0.5].Qh.nxvalue

punch_template = standard_lu(standard_x, standard_y, standard_z, 
                             (radius_l, radius_k, radius_h))


# # MATERN INTERPOLATION ########################################

# Set parameters
# epsilon: regularization (smoothness) parameter
# m: order parameter

# we will use a punch radius of 0.2
epsilon = 0.0
m = 2

starttime = timeit.default_timer()

z3d_restored = Main.interp_nexus(x, x2, x3, z3d, punch_template, 
                                          epsilon, m)

print("Time taken for Matern interpolation with punch radius 0.2, epsilon = " +
      str(epsilon) + " m = " + str(m) + ": " ,
      timeit.default_timer() - starttime)

# Laplace INTERPOLATION ########################################

starttime = timeit.default_timer()

z3d_restored_laplace = Main.interp_nexus(x, x2, x3, z3d, punch_template, epsilon, 1)

print("Time taken for Laplace interpolation with punch radius 0.2:", 
      timeit.default_timer() - starttime)

# PLOTTING COMMANDS (MATPLOTLIB) ###########################################

# Taking a 1D slice

# Index in x and y
# idx = 60
# idy = 10

# Find the maximum of the data on the slice common to both z3d_copy and z3d
# (are these different?) and add 10 for good measure
# max1  = np.max(z3d_copy[:,idy,idx])
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
# ## Save the Matern and Laplace Interpolated data to an .nxs file in the save
# directory

expt_data = nxload(filename)['entry']

root = NXroot(NXentry())
stdinterp = NXfield(symmetrize(z3d_restored[25:351, 25:451, 25:451]), name='sphere_punch_matern_interp_data')
root.entry.sphere_matern_data = NXdata(stdinterp,
                        expt_data.symm_transform[-6.:6, -8.:8., -8.:8].nxaxes)

sphmat = save_data_dir + '/movo2_40_sphere_matern_data.nxs'

if os.path.exists(sphmat):
    os.remove(sphmat)
 
 
root.save(sphmat)
 
root = NXroot(NXentry())
stdinterp = NXfield(symmetrize(z3d_restored_laplace[0:326, 0:426, 0:426]), name='sphere_punch_laplace_interp_data')
root.entry.sphere_laplace_data = NXdata(stdinterp, expt_data.symm_transform[-6.:5.98, -8.:7.98, -8.:7.98].nxaxes)

sphlap = save_data_dir + '/movo2_40_sphere_laplace_data.nxs'

if os.path.exists(sphlap):
    os.remove(sphlap)


root.save(sphlap)

print("Files "+sphmat+ " and " + sphlap + " saved in: ", save_data_dir)

# # # EOF ########################################################################
