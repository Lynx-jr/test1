from mpi4py import MPI
import scipy.stats as sts
import numpy as np
import time
def sim_healthshocks(lives):

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()
  t0 = time.time()
  
  # Set model parameters
  rho = 0.5
  mu = 3.0
  sigma = 1.0
  z_0 = mu

  # Set simulation parameters, draw all idiosyncratic random shocks,
  # and create empty containers
  S = int(lives/size) # Set the number of lives to simulate
  T = int(4160) # Set the number of periods for each simulation
  np.random.seed(26)
  eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S))
  z_mat = np.zeros((T, S))

  for s_ind in range(S):
    z_tm1 = z_0
  for t_ind in range(T):
    e_t = eps_mat[t_ind, s_ind]
    z_t = rho * z_tm1 + (1 - rho) * mu + e_t
    z_mat[t_ind, s_ind] = z_t
    z_tm1 = z_t

  # Gather all simulation arrays to buffer of expected size/dtype on rank 0
  z_all = None
  if rank == 0:
      z_all = np.empty([S*size, 4160], dtype = 'float')
  comm.Gather(sendbuf = z_mat, recvbuf = z_all, root = 0)

  if rank == 0:
    time_e = time.time() - t0 
    print("Simulated %d Lifetime Heathshocks in: %f seconds on %d MPI processes"
      % (lives, time_e, size))
  
def main():
    sim_healthshocks(1000)

if __name__ == '__main__':
    main()
