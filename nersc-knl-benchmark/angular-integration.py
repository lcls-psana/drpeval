from __future__ import division, print_function, absolute_import

import sys
import numpy as np
from numba import jit
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 1024

@jit(nopython=True)
def bincount(x, weights):
    N = x.shape[0]
    hist = np.zeros(725, dtype=np.float32) #FIXME
    for i in range(N):
        hist[x[i]] += weights[i]
    return hist

class AngularIntegrator:
    def __init__(self, xcenter, ycenter, radial_bins):
        x = np.arange(N) - xcenter
        y = np.arange(N) - ycenter
        X, Y = np.meshgrid(x, y)
        radius = np.sqrt(X**2 + Y**2).astype(np.float32)
        self.bin_indices = np.digitize(radius, radial_bins).astype(np.int32).reshape(-1)
        self.norm = np.bincount(self.bin_indices)

        # call bincount once to warm up jit
        _ = bincount(self.bin_indices, radius.reshape(-1))

    def __call__(self, img):
        #hist = np.bincount(self.bin_indices, img.reshape(-1))
        hist = bincount(self.bin_indices, img.reshape(-1))
        return hist / self.norm

nimages = 30
images = np.random.rand(nimages, N, N).astype(np.float32)
bins = np.linspace(1.0, N-100, N-100, dtype=np.float32)
integrator = AngularIntegrator(N/2, N/2, bins)

comm.Barrier()
start = MPI.Wtime()

for i in range(nimages):
    radial_profile = integrator(images[i])

comm.Barrier()
end = MPI.Wtime()

if rank == 0:
    print('%.2f GB' %(1024*1024*4*nimages*size / 1024**3))
    print(end - start, '%.1e images / sec ' %(nimages*size / (end - start)))
    print('%.2f GB/s' %(1024*1024*4*nimages*size / (end - start) / 1024**3))
    with open(sys.argv[1], 'a') as fh:
        fh.write('%d %f\n' %(size, end - start))
