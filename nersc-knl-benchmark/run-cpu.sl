#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -p regular
#SBATCH -t 02:00:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

module load python

echo 'haswell'

#run the application:
python run.py -t haswell -o results-python-haswell.txt -l python
python run.py -t haswell -o results-cpp-haswell.txt -l c++
