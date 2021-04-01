import numpy as np
import matplotlib.pyplot as plt
from   scipy.interpolate import Rbf
from   nexusformat.nexus import *

from skimage.filters import laplace
from skimage.color import label2rgb
from scipy import ndimage as ndi
from skimage import segmentation

from   pathlib import Path
home = str(Path.home())

from skimage import data
from skimage.exposure import histogram

import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib import animation, rc
from IPython.display import HTML

import scipy.signal
from   astropy.convolution import convolve, Kernel, Gaussian1DKernel
import math
import   scipy.ndimage 

import math, timeit

# Note: the julia package Laplacians.jl was forked by me and modified. It lives at 
# https://bitbucket.org/clhaley/Laplacians.jl
# If you use the registered package at
# https://github.com/danspielman/Laplacians.jl
# you will get a conflict between the matplotlib plotting software and the julia
# plotting software that is a package dependency, and we don't want that. My package above is identical except that
# it resolves this problem. 
from julia import Julia
julia = Julia(compiled_modules=False)
from julia import Main
julia.eval("@eval Main import Base.MainInclude: include")

def flipaxis(A,i):
    Aprime=np.swapaxes(np.swapaxes(A,0,i)[::-1],0,i)
    return Aprime

def getbraggs(x,dxx):
    return np.nonzero((x-np.rint(x))**2<(dxx/2)**2)[0]

def getstencil(add):
    m = 2*add+1
    sten = np.zeros((m,m,m))
    sten[add,add,add] = 1.0
    s = np.array([[[0,0,0],[0,1,0],[0,0,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,0,0],[0,1,0],[0,0,0]]])
    for i in range(add):
        sten = scipy.ndimage.convolve(sten, s)
    return np.where(sten > 0.0, 1.0, 0.0) 

# Function that wraps the julia code for the laplacian interpolation
def laplacian_fill(z3d, new_punch_locs):
    res = julia.laplacians_julia(z3d.shape, np.where(new_punch_locs.flatten()==0)[0], z3d[new_punch_locs==0].flatten())
    return z3d*(1-new_punch_locs)+res*new_punch_locs

# Function that wraps the julia code for the matern interpolation
def matern_fill(z3d, new_p, epsilon, m):
    res = julia.maternDE_julia(z3d.shape, np.where(new_p.flatten()==0)[0], z3d[new_p==0].flatten(), epsilon, m)
    return z3d*(1-new_p)+res*new_p

# Peeling code: take off an extra (add) number of pixels around each Bragg peak
def peel(data, add):
    sten = getstencil(add)
    return np.where(np.isnan(scipy.ndimage.convolve(data, sten, mode = 'constant', cval = 0.0)) == 1, np.nan, 1.0)*data

def standard_punch(x,x2,x3,z3d,rad):
    L,K,H = np.meshgrid(x,x2,x3,indexing='ij')
    punch_locs = np.where(((L-np.rint(L))**2/(1.**2)+(K-np.rint(K))**2/(1.5**2)+(H-np.rint(H))**2/(1.5**2))<(rad)**2,np.nan*np.ones(np.shape(z3d)),np.ones(np.shape(z3d)))
    # Perform the punch
    punched   = np.multiply(z3d, punch_locs)
    nan_locs  = np.where(np.isnan(punched)==1,np.ones(np.shape(z3d)),np.zeros(np.shape(z3d)))
    return punched, nan_locs

# Define the symmetrizing operation from the first quadrant
# Inputs: res assumes you are going from -1 l.u. to the max number of l.u.
#         scl_{k,i,j} 
def symmetrize(res, Nh, Nk, Nl, scl_k, scl_i, scl_j):
    # Convolultion happens over a single octant of the dataset
    vvals=np.zeros((Nh,Nk,Nl))
    N2h = (Nh-1)/2  
    N2k = (Nk-1)/2  
    N2l = (Nl-1)/2  
    vvals[(N2h-scl_j):Nh,(N2k-scl_k):Nk,(N2l-scl_i):Nl]=res
    # res = data.entry.transform[-1.:10.,-1.:50.,-1.:10.].data.nxdata
    vvals[N2h:Nh,N2k:Nk,0:(N2l+1)]=flipaxis(vvals[N2h:Nh,N2k:Nk,N2l:Nl],2)
    vvals[N2h:Nh,0:(N2k+1),0:Nl]=flipaxis(vvals[N2h:Nh,N2k:Nk,0:Nl],1)
    vvals[0:(N2h+1),0:Nk,0:Nl]=flipaxis(vvals[N2h:Nh,0:Nk,0:Nl],0)
    # no background subtraction 
    return vvals[0:(Nh-1),0:(Nk-1),0:(Nl-1)]

def _round_up_to_odd_integer(value):
    i = int(math.ceil(value))
    if i % 2 == 0:
        return i + 1
    else:
        return i

class Gaussian3DKernel(Kernel):

    _separable = True
    _is_bool = False

    def __init__(self, stddev, **kwargs):
        x = np.linspace(-15., 15., 17)
        y = np.linspace(-15., 15., 17)
        z = np.linspace(-15., 15., 17)
        X,Y,Z = np.meshgrid(x,y,z)
        array = np.exp(-(X**2+Y**2+Z**2)/(2*stddev**2))
        self._default_size = _round_up_to_odd_integer(8 * stddev)
        super(Gaussian3DKernel, self).__init__(array)
        self.normalize()
        self._truncation = np.abs(1. - self._array.sum())

gkernal = Gaussian3DKernel(2)

def standard_fill(z3d, punched, nan_locs):
    vvals = convolve(punched,gkernal)
    res   = z3d*(1-nan_locs)+vvals*nan_locs
    return res