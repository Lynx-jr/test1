#!/bin/bash
#SBATCH --job-name=question_3_c   # job name
#SBATCH --output=gpu_question_3c.out # output log file
#SBATCH --error=gpu.err2  # error file
#SBATCH --time=00:05:00  # 5 minutes of wall time
#SBATCH --nodes=1        # 1 GPU node
#SBATCH --partition=gpu2 # GPU2 partition
#SBATCH --ntasks=1       # 1 CPU core to drive GPU
#SBATCH --gres=gpu:1     # Request 1 GPU

module load cuda
module load python/anaconda-2019.03

python3 ./question_3_c.py
