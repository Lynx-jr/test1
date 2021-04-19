
import rasterio
import pyopencl as cl
import pyopencl.tools as cltools
import pyopencl.array as cl_array
import numpy as np
import time

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# Import bands as separate images; in /project2/macs30123 on RCC
band4 = rasterio.open('LC08_B4.tif') #red
band5 = rasterio.open('LC08_B5.tif') #nir
# Convert nir and red objects to float64 arrays
red = band4.read(1).astype('float64')
nir = band5.read(1).astype('float64')
# NDVI calculation
t0 = time.time()
ndvi = (nir - red) / (nir + red)
time_e1 = time.time() - t0

# Opencl
t0 = time.time()
red_dev = cl_array.to_device(queue,red)
nir_dev = cl_array.to_device(queue,nir)
ndvi_dev = (red_dev - nir_dev)/(red_dev + nir_dev)

ndvi = ndvi_dev.get()
time_e2 = time.time() - t0
print('OpenCL array implementation took %f seconds. Previous one took %f seconds.'
 % (time_e2,time_e1))
