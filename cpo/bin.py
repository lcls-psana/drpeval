from numba import float32, jit, int64
import numpy as np
import time
import sys

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

niter=int(sys.argv[1])

def rebin(calib,binned_calib):
    npanel,nrow,ncol = calib.shape
    for i in range(npanel):
        for j in range(0,nrow,2):
            for k in range(0,ncol,2):
                val = (calib[i,j,k]+calib[i,j+1,k]+calib[i,j,k+1]+calib[i,j+1,k+1])*0.25
                binned_calib[i,j,k]=val
                if j+1 < nrow: binned_calib[i,j+1,k]=val
                if k+1 < ncol: 
                    binned_calib[i,j,k+1]=val
                    if j+1 < nrow:
                        binned_calib[i,j+1,k+1]=val
    return binned_calib

fastrebin = jit(float32[:,:,:](float32[:,:,:],float32[:,:,:]))(rebin)

calib = np.ones([niter,32,185,388],dtype=np.float32)
binned_calib = np.ones([32,185,388],dtype=np.float32)
tstart=time.time()
comm.Barrier()
for i in range(niter):
    fastrebin(calib[i],binned_calib)
print (time.time()-tstart)/niter
