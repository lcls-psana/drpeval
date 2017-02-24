Running benchmarks at Cori@NERSC comparing 1 haswell node (2 * 16 core Intel Haswell processor)
to 1 Intel Phi node (1 Intel Phi processor 64 core)

to run benchmarks do:
* ./build.sh
* sbatch run-knl.sl
* sbatch run-cpu.sl
* sqs # to monitor progress in the queueing system
* run the benchmark-analysis.ipynb notebook to look at the results

### Conclusion
* knl node and haswell node show comparable performance
* knl is faster using the fast MCDRAM memory and a little bit slower using the normal DDR4 memory
* python and cpp performance is also very similar, after tweaking the python implementation
* np.bincount casts input weights to double. This costs a factor of 3 in performance compared to a numba implementation without the type conversion.
