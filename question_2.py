from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import time

# Set model parameters
rho_array_total = np.linspace(-0.95, 0.95, 200)
mu = 3.0
sigma = 1.0
z_0 = mu

S = 1000 # Set the number of lives to simulate
T = int(4160) # Set the number of periods for each simulation np.random.seed(25)
eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S))
number_rho = 200


def sim(number_rho):
    # Get rank of process and overall size of communicator:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Start time:
    t0 = time.time()

    # Evenly distribute number of rho runs across processes
    N = int(number_rho / size) #200/20 = 10rhos/core

    average_period = np.zeros(N)
    i = 0
    for rho_ind in range(N*rank, N*(rank+1)):
      rho = rho_array_total[rho_ind]
      z_mat = np.zeros((T, S))
      period = np.zeros(S)
      # Simulate S random walks on each MPI Process
      for s_ind in range(S):
        z_tm1 = z_0
        for t_ind in range(T):
            e_t = eps_mat[t_ind, s_ind]
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t
            if z_t < 0:
              period[s_ind] = t_ind + 1
              break
            z_mat[t_ind, s_ind] = z_t
            z_tm1 = z_t
        if period[s_ind] == 0:
          period[s_ind] = T
      average_period[i]= np.mean(period)
      i = i + 1

    rhos_average_period = np.empty(number_rho)
    if rank == 0:
        rhos_average_period = np.empty(number_rho)
    comm.Gather(sendbuf = average_period, recvbuf = rhos_average_period, root = 0)

    if rank == 0:
      optimal_rho_ind = np.where(rhos_average_period == np.max(rhos_average_period))
      optimal_rho = rho_array_total[optimal_rho_ind]
      time_elapsed = time.time() - t0
      print("The computational time is", time_elapsed)
      print(rhos_average_period)
      #b
      plt.plot(rho_array_total, rhos_average_period)
      plt.title("Corresponding Average Period for Negative Health with Different Rhos")
      plt.xlabel('Rho')
      plt.ylabel('Average Period')
      plt.savefig("plot_q2.png")
      #c
      print("The optimal rho is", optimal_rho[0])
      print("The corresponding avarege period to optimal rho is",np.max(rhos_average_period))
        
def main():
    sim(200)
    
if __name__ == '__main__':
    main()
