#!/bin/bash
mkdir $1
bsub -n 2 -o $1/%J.log -q psfehhiprioq mpirun code1.sh $1
