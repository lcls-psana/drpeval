import h5py
import numpy as np
from ImgAlgos.PyAlgos import PyAlgos
import timeit


data_file = "/reg/d/psdm/cxi/cxitut13/scratch/cpo/test_bigh5.h5"
# data_file = "small_test.h5"
events = 31
fsrc = h5py.File(data_file,'r')
cspad = fsrc['cspad']
alg = PyAlgos()
alg.set_peak_selection_pars(npix_min=2, npix_max=50, amax_thr=10, atot_thr=20, son_min=5)
# hdr = '\nSeg  Row  Col  Npix    Amptot'
# fmt = '%3d %4d %4d  %4d  %8.1f'

hdr = 'Seg  Row  Col  Npix      Amax      Atot   rcent   ccent rsigma  csigma '+\
      'rmin rmax cmin cmax    bkgd     rms     son\n'
fmt = '%3d %4d %4d  %4d  %8.1f  %8.1f  %6.1f  %6.1f %6.2f  %6.2f %4d %4d %4d %4d  %6.2f  %6.2f  %6.2f\n'


# fw = open("peakFindResult_python", 'w')
total_count = np.array(0,'i')

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print("peakFinderRankSize {0}".format(size))
# print("size:{0},rank:{1}".format(size,rank))
counter = np.array(0,'i')
step = events / size
st = rank * step
ed = min((rank+1)*step, events)
print("task {0} start from {1} to {2}".format(rank,st,ed))
cspad = cspad[st:ed,:,:,:]

comm.Barrier()

start = timeit.default_timer()
for i in range(ed-st):
	# fw.write(hdr)
	# if i % size != rank: continue
	peaks = alg.peak_finder_v4r2(cspad[i,:,:,:],thr_low=10, thr_high=150, rank=4, r0=5, dr=0.05)
	counter += 1
	# for peak in peaks :
	#     seg,row,col,npix,amax,atot,rcent,ccent,rsigma,csigma,\
	#     rmin,rmax,cmin,cmax,bkgd,rms,son = peak[0:17]
	     
	#     fw.write( fmt % (seg, row, col, npix, amax, atot, rcent, ccent, rsigma, csigma,\
	#                  rmin, rmax, cmin, cmax, bkgd, rms, son))
stop = timeit.default_timer()
print("peakFinderTask: rank:{0}, event:{3}, start:{1:.4f}, end:{2:.4f}".format(rank,start,stop,i+st))
print("peakFinderPyTask:{0}".format(stop-start))
# comm.Reduce(counter,total_count)

# comm.Barrier()
if rank == 0:
    print("size:{0}",format(size))
    print("counter: {0}".format(total_count))
# fw.close()




