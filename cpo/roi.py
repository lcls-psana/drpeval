from numba import float32, jit, int64
import numpy as np
import time
import sys

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

niter=int(sys.argv[1])

def fastroi_py(calib,roi):
    nrow,ncol = roi.shape
    nabovethresh = 0
    for i in range(0,nrow):
        for j in range(0,ncol):
            roi[i,j]=calib[i,j]
    return 0

fastroi = jit(int64(float32[:,:],float32[:,:]))(fastroi_py)

roi = np.zeros([300,300],dtype=np.float32)
calib = np.ones([niter,1000,1000],dtype=np.float32)
tstart=time.time()
comm.Barrier()
for i in range(niter):
    fastroi(calib[i],roi)
print (time.time()-tstart)/niter
