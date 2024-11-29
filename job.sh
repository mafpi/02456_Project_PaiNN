#!/bin/bash
#BSUB -q gpua100
#BSUB -J painn5layers_
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=512MB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 15:00
#BSUB -B 
#BSUB -N 
#BSUB -o painn5layers_%J.out
#BSUB -e painn5layers_%J.err

cd /dtu/blackhole/00/202496/02456_Project_PaiNN

hpcintrogpush

python minimal_example.py

