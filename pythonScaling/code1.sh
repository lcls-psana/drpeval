#!/bin/bash
#time sleep 2
#time python mpiReduce.py
#time python test.py
#strace -c -o strace.out.$$ python /reg/neh/home/yoon82/examplePython/mpiReduce.py
strace -f -tt -o $@/strace.out.$$ python /reg/neh/home/yoon82/Software/drpeval/pythonScaling/test.py
