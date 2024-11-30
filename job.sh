#!/bin/bash
#BSUB -q gpuv100
#BSUB -J painn3layers
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 10:00
#BSUB -o painn3layers%J.out
#BSUB -e painn3layers%J.err


# Initialize Python environment
hpcintrogpush

python minimal_example.py