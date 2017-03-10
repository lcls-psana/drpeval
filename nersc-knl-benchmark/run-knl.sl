#!/bin/bash
#SBATCH -N 1
#SBATCH -C knl,quad,flat
#SBATCH -p regular
#SBATCH -t 00:40:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

module load python

#run the application:
#python run.py -t knl -o results-python-knl.txt -l python
python /global/homes/w/weninc/drpeval/nersc-knl-benchmark/run.py -t knl -o results-cpp-knl-flat-vec.txt -l c++
