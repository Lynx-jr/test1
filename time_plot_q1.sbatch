#!/bin/bash

#SBATCH --job-name=time_plot_q1
#SBATCH --output=time_plot_q1.out
#SBATCH --ntasks=20
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr

# Load Python and MPI modules
module load python/anaconda-2019.03
module load intelmpi/2018.2.199+intel-18.0

mpirun -n 1 python3 time_plot_q1.py >> time_plot_q1.out
