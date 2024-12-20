#!/bin/bash
#BSUB -q gpua100
#BSUB -J painn5layers_
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=10GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 15:00
#BSUB -B 
#BSUB -N 
#BSUB -o painn5layers_%J.out
#BSUB -e painn5layers_%J.err

cd /dtu/blackhole/00/202496/02456_Project_PaiNN

python minimal_example.py --num_message_passing_layers 5