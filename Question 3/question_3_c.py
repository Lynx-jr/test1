import numpy as np
import matplotlib.pyplot as plt
import time
import rasterio

import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.tools as cltools
from pyopencl.elementwise import ElementwiseKernel


def sim_tilt():
  band4 = rasterio.open('/project2/macs30123/landsat8/LC08_B4.tif') #red
  band5 = rasterio.open('/project2/macs30123/landsat8/LC08_B5.tif') #nir
  red = band4.read(1).astype('float64')
  nir = band5.read(1).astype('float64')
  red_10 = np.tile(red, 10)
  nir_10 = np.tile(nir, 10)
  #cpu
  t0 = time.time()
  nvdi_cpu = (nir_10 - red_10) / (nir_10 + red_10)
  time_cpu = time.time() - t0
  #gpu
  t1 = time.time()
  ctx = cl.create_some_context()
  queue = cl.CommandQueue(ctx) 
  red_gpu = cl_array.to_device(queue, red_10)
  nir_gpu = cl_array.to_device(queue, nir_10)
  nvdi_formula = ElementwiseKernel(ctx,"double *x, double *y, double *nvdi","nvdi[i] = (x[i] - y[i]) / (x[i] + y[i])")
  nvdi_gpu = cl.array.empty_like(nir_gpu)
  nvdi_formula(nir_gpu, red_gpu, nvdi_gpu)
  nvdi_gpu_new = nvdi_gpu.get()
  time_gpu = time.time() - t1

  print("The time of CPU computation for 10 times scene is", time_cpu)
  print('The time of GPU computation for 10 times scene is', time_gpu)

  red_20 = np.tile(red, 20)
  nir_20 = np.tile(nir, 20)
  #cpu
  t0 = time.time()
  nvdi_cpu = (nir_20 - red_20) / (nir_20 + red_20)
  time_cpu = time.time() - t0
  #gpu
  t1 = time.time()
  ctx = cl.create_some_context()
  queue = cl.CommandQueue(ctx) 
  red_gpu = cl_array.to_device(queue, red_20)
  nir_gpu = cl_array.to_device(queue, nir_20)
  nvdi_formula = ElementwiseKernel(ctx,"double *x, double *y, double *nvdi","nvdi[i] = (x[i] - y[i]) / (x[i] + y[i])")
  nvdi_gpu = cl.array.empty_like(nir_gpu)
  nvdi_formula(nir_gpu, red_gpu, nvdi_gpu)
  nvdi_gpu_new = nvdi_gpu.get()
  time_gpu = time.time() - t1

  print("The time of CPU computation for 20 times scene is", time_cpu)
  print('The time of GPU computation for 20 times scene is', time_gpu)

def main():
  sim_tilt()

if __name__ == '__main__':
  main()
