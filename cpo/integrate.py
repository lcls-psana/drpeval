import numpy as np
import time
import sys

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

niter = int(sys.argv[1])

sum = np.zeros([32,185,388],dtype=np.float32)
img = np.ones([niter,32,185,388],dtype=np.float32)
comm.Barrier()
tstart=time.time()
for i in range(niter):
    sum += img[i,:,:,:]
print (time.time()-tstart)/niter,sum[0,0,0]
