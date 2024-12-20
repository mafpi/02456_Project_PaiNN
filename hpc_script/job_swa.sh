#!/bin/bash
#BSUB -q gpua100
#BSUB -J swa_painn_
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=10GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 15:00
#BSUB -B 
#BSUB -N 
#BSUB -o swa_painn_%J.out
#BSUB -e swa_painn_%J.err

cd /dtu/blackhole/00/202496/02456_Project_PaiNN

python swa_multipleLR.py --num_message_passing_layers 3 --num_epochs 50