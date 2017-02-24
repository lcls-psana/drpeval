from __future__ import division, absolute_import, print_function

import os
import sys
import time
import shlex
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t','--target', help='Target to run on. Either knl or haswell',
                    required=True, type=str)
parser.add_argument('-o','--output', help='Output file name for benchmark results',
                    required=True, type=str)

parser.add_argument('-l','--language', help='Run benchmark in Python: python or C++: c++',
                    required=True, type=str)

args = parser.parse_args()

if args.target == 'haswell':
    # haswell @ cori
    def get_cmd(nproc, output_file):
        if nproc <= 32:
            spread = 2
        else:
            spread = 1
        if args.language == 'c++':
            return 'srun -n %d -c %d --cpu_bind=cores /global/homes/w/weninc/haswell.out %s' %(nproc, spread, output_file)
        elif args.language == 'python':
            return 'srun -n %d -c %d --cpu_bind=cores python /global/homes/w/weninc/angular-integration.py %s' %(nproc, spread, output_file)

    nprocs = list(range(2, 34, 2))
    nprocs.insert(0, 1)
    nprocs.append(64)

elif args.target == 'knl':
    # knl @ cori
    def get_cmd(nproc, output_file):
        if nproc <= 64:
            spread = 4
        elif nproc <= 128:
            spread = 2
        else:
            spread = 1
        if args.language == 'c++':
            return 'srun -n %d -c %d --cpu_bind=cores numactl -p 1 /global/homes/w/weninc/knl.out %s' %(nproc, spread, output_file)
        elif args.language == 'python':
            return 'srun -n %d -c %d --cpu_bind=cores numactl -p 1 python /global/homes/w/weninc/angular-integration.py %s' %(nproc, spread, output_file)
            # numactl -p 1

    nprocs = list(range(2, 66, 2))
    nprocs.insert(0, 1)
    nprocs.append(128)
    nprocs.append(256)

else:
    print('Wrong target!')
    sys.exit(-1)

def run(output_file, nprocs):
    if os.path.isfile(output_file):
        os.remove(output_file)

    for nproc in nprocs:
        print('nproc:', nproc)
        cmd = shlex.split(get_cmd(nproc, output_file))
        print(cmd)
        for j in range(3):
            subprocess.call(cmd)
            time.sleep(1)

run(args.output, nprocs)
