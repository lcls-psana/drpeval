from numba import float32, jit, int64
import numpy as np
import time
import sys

niter=int(sys.argv[1])

def fasttrig_py(bkgd,calib,thresh):
    npanel,nrow,ncol = calib.shape
    nabovethresh = 0
    for i in range(npanel):
        for j in range(0,nrow):
            for k in range(0,ncol):
                val = calib[i,j,k]-bkgd[i,j,k]
                if val>thresh: nabovethresh+=1
    return nabovethresh

fasttrig = jit(int64(float32[:,:,:],float32[:,:,:],float32))(fasttrig_py)

bkgd = np.zeros([32,185,388],dtype=np.float32)
calib = np.zeros_like(bkgd)
tstart=time.time()
for i in range(niter):
    fasttrig(bkgd,calib,10.1)
print (time.time()-tstart)/niter
