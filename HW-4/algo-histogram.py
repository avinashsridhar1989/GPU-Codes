#!/usr/bin/env python

"""
Histogram Calculation for N=R*C bytes image where R and C are byte dimensions.
Pixels are 8 bit values from 0 - 255
By Avinash Sridhar (as4626@columbia.edu)

Note: Open README.md to see brief details of algorithms
"""

import time

import pyopencl as cl
import pyopencl.array
import numpy as np

# This is the basic histogram algorithm in python. Provided by TA.
def hist(x):
    bins = np.zeros(256, np.uint32)
    for v in x.flat:
        bins[v] += 1
    return bins

# Select the desired OpenCL platform; you shouldn't need to change this:
NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
    if platform.name == NAME:
        devs = platform.get_devices()

# To check which GPU is being used
print devs

# Set up a command queue:
ctx = cl.Context(devs)
queue = cl.CommandQueue(ctx)

# Create input image containing 8-bit pixels; the image contains N = R*C bytes;
# When reading image from /opt/data/random.dat please use R=1000*100 and C=1000*10 for 10^9 bytes of data size
P = 32
R = P*32
C = P*32
N = R*C

# The below command has been hased as we are reading image from /opt/data/random
# Please use below numpy command for image generations upto 500 MB.
img = np.random.randint(0, 255, N).astype(np.uint8).reshape(R, C)
#print img

# You can create a mapped 1d array to an existing file of bytes using
# Ensure the below image is executed with mode = 'r' as we are reading from root folder.

#img = np.memmap('/opt/data/random.dat', dtype=np.uint8, mode = 'r')

# Below is the kernel implemented by me. Commends included within kernel for brief explanations.
my_kernel = """
__kernel void algo(__global unsigned char* img, __global unsigned int* final_bin, const int size) {

    // Each temp[localId] will map to a histogram value as we will use 256 work items per work groupId
    __local unsigned int temp[256];
    unsigned int localId   = get_local_id(0);
    unsigned int globalId  = get_global_id(0);
    unsigned int groupId   = get_group_id(0);
    unsigned int localSize = get_local_size(0);
    
    // We will initialize all work items of a work group to zero. Hence we need synchronization.
    temp[localId] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // We calculate values of i and offset, and iterate through the entire image. As we iterate through entire image, we will jump
    // in increments of offset, skipping 256 pixels in a go. This way we will obtain histogram values in chunks of 256. Note that a
    // barrier synchronization is required at end of while loop to ensure that there is synchronization. Also we
    // proceed to the next step only once all work groups have completed the while loop exeuction.
    // Atomic addition is done to prevent race conditions between threads.
    int i = localId + groupId * localSize;
    int offset = localSize * get_num_groups(0);
    while ( i < size)
    {
        atomic_add(&temp[img[i]], 1);
        i += offset;
    }
  barrier(CLK_LOCAL_MEM_FENCE);

  // We are finally storing the obtained histogram value from temp[localId] to final_bin[localId]. Note this gives right histogram
  // results as localId is mapped to histogram bin values (0-256)
  atomic_add(&final_bin[localId], temp[localId]);

}
"""

func = cl.Program(ctx, """
__kernel void func(__global unsigned char *img, __global unsigned int *bins,
                   const unsigned int P) {
    unsigned int i = get_global_id(0);
    unsigned int k;
    volatile __local unsigned char bins_loc[256];

    for (k=0; k<256; k++)
        bins_loc[k] = 0;
    for (k=0; k<P; k++)
        ++bins_loc[img[i*P+k]];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (k=0; k<256; k++)
        atomic_add(&bins[k], bins_loc[k]);
}
""").build().func

# PLEASE hash the below code during execution of larger size images, as python will take a long time to iterate in the for loops
#start = time.time()
#h_py = hist(img)
#time_python = time.time() - start
#print h_py
#

func.set_scalar_arg_dtypes([None, None, np.uint32])
img_gpu = cl.array.to_device(queue, img)
bin_gpu = cl.array.zeros(queue, 256, np.uint32)
start = time.time()
var1 = func(queue, (N/32,), (1,), img_gpu.data, bin_gpu.data, np.uint32(P))
var1.wait()
#time_basic_algo = time.time() - start
bin_basic_opencl = bin_gpu.get()
print bin_basic_opencl

mf = cl.mem_flags

prg = cl.Program(ctx, my_kernel).build()
img_gpu = cl.array.to_device(queue, img)
bin_gpu_new = cl.array.zeros(queue, 256, np.uint32)
start = time.time()
var2 = prg.algo(queue, (N, ), (256, ), img_gpu.data, bin_gpu_new.data, np.uint32(N))
var2.wait()
#time_my_algo = time.time() - start
bin_my_opencl = bin_gpu_new.get()
print bin_my_opencl

#print "--Basic Algo provided by TA and Python check execution times-- ", np.allclose(h_py, bin_basic_opencl)
print "--Basic Algo provided by TA and my algorithm check execution times-- ", np.allclose(bin_basic_opencl, bin_my_opencl)

loops = 3
times = []
# Calculate execution time for basic algo
func.set_scalar_arg_dtypes([None, None, np.uint32])
img_gpu = cl.array.to_device(queue, img)
for i in range(loops):
    bin_gpu = cl.array.zeros(queue, 256, np.uint32)
    start = time.time()
    var1 = func(queue, (N/32,), (1,), img_gpu.data, bin_gpu.data, np.uint32(P))
    var1.wait()
    times.append(time.time() - start)
time_basic_algo = np.average(times)
mf = cl.mem_flags

times = []
# Calculate execution time for my algo
prg = cl.Program(ctx, my_kernel).build()
img_gpu = cl.array.to_device(queue, img)
for i in range(loops):
    bin_gpu_new = cl.array.zeros(queue, 256, np.uint32)
    start = time.time()
    var2 = prg.algo(queue, (N, ), (256, ), img_gpu.data, bin_gpu_new.data, np.uint32(N))
    var2.wait()
    times.append(time.time() - start)
time_my_algo = np.average(times)

#print "PYTHON EXECUTION TIME: ", time_python
print "BASIC ALGO EXECUTION TIME: ", time_basic_algo
print "MY ALGO EXECUTION TIME: ", time_my_algo





