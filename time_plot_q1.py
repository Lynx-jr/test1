from mpi4py import MPI
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt

def plot_time():
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  if rank == 0:
    time = np.loadtxt("time_q1.txt")
  
  num_cores = np.arange(20)+1
  plt.plot(num_cores, time)  
  plt.title("Computation Time with Different Number of Cores")
  plt.xlabel("Number of cores") 
  plt.ylabel("Computation time")
  plt.savefig("plot_q1.png")

def main():
  plot_time()

if __name__ == '__main__':
  main()
