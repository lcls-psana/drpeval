from numba import float32, jit
import numpy as np

def fasttrig_py(bkgd,calib,thresh):
    npanel,nrow,ncol = calib.shape
    nabovethresh = 0
    for i in range(npanel):
        for j in range(0,nrow):
            for k in range(0,ncol):
                val = calib[i,j,k]-bkgd[i,j,k]
                if val>thresh: nabovethresh+=1
    return nabovethresh

fasttrig = jit(int(float32[:,:,:],float32[:,:,:]))(fasttrig_py)

bkgd = np.zeros([32,185,388])
calib = np.zeros_like(bkgd)
print fasttrig(bkdg,calib,10)
