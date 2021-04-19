from mpi4py import MPI
import scipy.stats as sts
import numpy as np
import time
import matplotlib.pyplot as plt

t0 = time.time()
S = int(1000) # Set the number of lives to simulate
T = int(4160) # Set the number of periods for each simulation
mu = 3.0
sigma = 1.0
z_0 = mu
np.random.seed(27)
rhos = np.linspace(-0.95,0.95,num=200)
eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T,S))
z_mat = np.zeros((T, S))
rho_mat = np.zeros((10,2))

def gsearch(lives):

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()
  rho_used = rhos[range(10*rank,10*rank+10)]
  bestavg = 0
  bestrho = 0


  for rho_i in range(10):
    tdeath = [4160] * lives
    rho = rho_used[rho_i]

    for s_ind in range(lives):
      z_tm1 = z_0

      for t_ind in range(T):
        e_t = eps_mat[t_ind, s_ind]
        z_t = rho * z_tm1 + (1 - rho) * mu + e_t
        if z_t <= 0:
          tdeath[s_ind] = t_ind
          break
        z_mat[t_ind, s_ind] = z_t
        z_tm1 = z_t

    avgtdeath = sum(tdeath)/lives
    rho_mat[rho_i,0] = rho
    rho_mat[rho_i,1] = avgtdeath

  # Gather all simulation arrays to buffer of expected size/dtype on rank 0
  rho_all = None
  if rank == 0:
      rho_all = np.empty([200, 2], dtype = 'float')
  comm.Gather(sendbuf = rho_mat, recvbuf = rho_all, root = 0)

  if rank == 0:
    best_t = max(rho_all[:,1])
    best_index = np.where(rho_all[:,1] == best_t)
    best_rho = rhos[best_index] 
    time_e = time.time() - t0

    print("Best rho is %f with terrible things happen at time %f on average. Done in %f seconds"
      % (best_rho, best_t, time_e))

    rho = rho_all[:,0]
    avgd = rho_all[:,1]
    plt.plot(rho,avgd)
    plt.savefig('Fig2.jpg')

def main():
    gsearch(1000)

if __name__ == '__main__':
    main()
