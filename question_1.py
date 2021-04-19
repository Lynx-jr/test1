from mpi4py import MPI
import numpy as np
import scipy.stats as sts
import time

# Set model parameters
rho = 0.5
mu = 3.0
sigma = 1.0
z_0 = mu
# Set simulation parameters, draw all idiosyncratic random shocks, # and create empty containers
S = 1000 # Set the number of lives to simulate
T = int(4160) # Set the number of periods for each simulation np.random.seed(25)
eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S))


def sim(S):
    # Get rank of process and overall size of communicator:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Start time:
    t0 = time.time()

    # Evenly distribute number of simulation runs across processes
    N = int(S / size)
    z_mat = np.zeros((T, N))
    eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, N))
    # Simulate N random walks on each MPI Process and specify as a NumPy Array
    for s_ind in range(N):
        z_tm1 = z_0
        for t_ind in range(T):
            e_t = eps_mat[t_ind, s_ind]
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t
            z_mat[t_ind, s_ind] = z_t
            z_tm1 = z_t

    # Gather all simulation arrays to buffer of expected size/dtype on rank 0
    z_mat_all = None
    if rank == 0:
        z_mat_all = np.empty([T, N * size], dtype = 'float')
    comm.Gather(sendbuf = z_mat, recvbuf = z_mat_all, root = 0)

    # Print/plot simulation results on rank 0
    if rank == 0:
        # Calculate time elapsed after computing mean and std
        time_elapsed = time.time() - t0
        
        with open('time_question_1.txt','a') as f:
            print(time_elapsed, file = f)
        
def main():
    sim(S = 1000)
    
if __name__ == '__main__':
    main()
