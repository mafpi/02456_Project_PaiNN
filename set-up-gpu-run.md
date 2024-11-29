How to setup the HPC with GPU runnning

1. Make sure you log in to the hpc and in the terminal you see: hpclogin1(sXXXXXX) $ 
2. run: linuxsh
3. run: cd $BLACKHOLE
4. make sure you are in the path: /dtu/blackhole/XX/sXXXXXX
5. in that location store your files, you need: (you can just copy paste the folder from the repo)
    - job.sh
    - minimal_example.py
    - src
        - __init__.py
        - data
            - __init__.py
            - AtomNeighbours.py
            - qm9.py
        - models
            - __init__.py
            - painn.py
            - post_processing.py

6. with this structure you have to make sure you are in the same directory as job.sh and minimal_example.py
7. define all testing conditions in minimal_example.py
7. run: bsub < job.sh
8. you get something like: jobXXXX submitted in queue.
9. run: bstat to see the state of the job and job id. (you will receive email once it started and once it finished).
10. if you submitted it wrong and want to stop it, run: bkill jobidXXXX
12. to see .err and .out files run: cat NAMEFILE


## Structure of job.sh
#!/bin/bash
#BSUB -q gpua100                  ->>>>>>>>>>>> name of queue, gpu queue, jobs need to be in a queue  \
#BSUB -J painn5layers_     ->>>>>>>>>>>>> this is the job id and some numbers, like : painn5layers_XXXXXX, you can see the numbers when running bstat or your email. \
#BSUB -n 8                   ->>>>>>>>>>>> number of cores \
#BSUB -R "span[hosts=1]"                     ->>>>>>>>>>>> just runnign in the same machine \
#BSUB -R "rusage[mem=512MB]"                     ->>>>>>>>>>>> requested each core to have at least 512MB of memory \
#BSUB -gpu "num=1:mode=exclusive_process"                          ->>>>>>>>>>>> just requesting the GPU \
#BSUB -W 15:00             ->>>>>>>>>>>> max hours to run \
#BSUB -B              ->>>>>>>>>>>> receive email at start \
#BSUB -N              ->>>>>>>>>>>> receive email at finish \
#BSUB -o painn5layers_%J.out                  ->>>>>>>>>>>> outuput of py code \
#BSUB -e painn5layers_%J.err                  ->>>>>>>>>>>> outuput of portantial errors/prints in console \
\
cd /dtu/blackhole/00/202496/02456_Project_PaiNN               ->>>>>>>>>>>> replace with your directory from the blackhole and wher you run your minimal_example.py \

hpcintrogpush           

python minimal_example.py

