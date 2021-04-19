#!/bin/bash

#SBATCH --job-name=mpi_question_1
#SBATCH --ntasks=210
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr

# Load Python and MPI modules
module load python/anaconda-2019.03
module load intelmpi/2018.2.199+intel-18.0

for i in {1..20}
do
  mpirun -n $i python3 question_1.py >> mpi_question_1.out
done

mpirun -n 1 python3 time_plot_q1.py >> time_plot_q1.out
