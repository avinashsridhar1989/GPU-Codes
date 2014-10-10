#!/usr/bin/env python

"""
Bandwidth Limited Function using PyOpenCL.
By Avinash Sridhar (as4626@columbia.edu)
"""

import time
import pyopencl as cl
import numpy as np

# Selecting OpenCL platform.
NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
    if platform.name == NAME:
        devs = platform.get_devices()

# Setting Command Queue.
ctx = cl.Context(devs)
queue = cl.CommandQueue(ctx)

# OpenCL Kernel in C.

# Below function i.e. "void bandlimfunc" does the following for every kernel :
# 1) i will obtain the global_id from host using get_global_id(0)
# 2) Initialize temp_val = 0
# 3) Iterate from z = 0 o count. count = len(a) = N and is passed from 
#    prg.bandlimfunc(queue, x.shape, None, a_buf, b_buf, y_buf, x_buf, np.int32(N))
# 4) Perform summation within for loop in kernel to increment temp_val
# 5) Store final value of temp_val to y[i]
kernel = """
__kernel void bandlimfunc(__global float* a, __global float* b, __global float* y , __global float* x, int count) {
    unsigned int i = get_global_id(0);
    float temp_val = 0;
    for (int z = 0; z < count; z++)
    {
    	temp_val = temp_val + a[z]*cos(((float)z + 1)*x[i]) + b[z]*sin(((float)z + 1)*x[i]);
    }
    y[i] = temp_val;
}
"""
# Load some random data to process. Note that setting the data type is
# important; if your data is stored using one type and your kernel expects a
# different type, your program might either produce the wrong results or fail to
# run.  Note that Numerical Python uses names for certain types that differ from
# those used in OpenCL. For example, np.float32 corresponds to the float type in
# OpenCL:
dx = 0.05
x_max = 6.0
x = np.arange(0.0, x_max, dx).astype(np.float32)

N = 4
a_max = 5
b_max = 7
a = np.random.randint(0, a_max, N).astype(np.float32)
b = np.random.randint(0, b_max, N).astype(np.float32)

# Creating two numpy arrays of size x and values 0.
# py_val is equivalent to y here, except that py_val will do the calculation in python
py_val = np.zeros_like(x)
y = np.zeros_like(x)

# You need to set the flags of the buffers you create properly; otherwise,
# you might not be able to read or write them as needed:
mf = cl.mem_flags

x_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
y_buf =cl.Buffer(ctx, mf.WRITE_ONLY, x.nbytes)

# Launch the kernel; notice that you must specify the global and locals to
# determine how many threads of execution are run. We can take advantage of Numpy to
# use the shape of one of the input arrays as the global size. Since our kernel
# only accesses the global work item ID, we simply set the local size to None:

#'N' is passed as np.int32() constant to kernel. We will iterate N times to perform sin and cos operations.
a_length = len(a)
x_length = len(x) 
prg = cl.Program(ctx, kernel).build()
prg.bandlimfunc(queue, x.shape, None, a_buf, b_buf, y_buf, x_buf, np.int32(N))

# Retrieve the results from the GPU:
cl.enqueue_copy(queue, y, y_buf)

print 'input (a):    ', a
print 'input (b):    ', b

for i in xrange(0, N):
	py_val += a[i]*np.cos((i+1)*x) + b[i]*np.sin((i+1)*x)

print 'numpy b/w limited function:  ', py_val
print 'opencl b/w limited function: ', y

# np.allclose() is a check function to see if Python and PyOpenCL values are exact matches. 
# I have used relative tolerance of 1e-02 as there might be minute differences after 4 to 5 decimal points.
print 'pyopencl and numpy are equal:        ', np.allclose(py_val, y, rtol = 1e-02)

# Here we compare the speed of performing the vector addition with Python and
# PyOpenCL. Since the execution speed of a snippet of code may vary slightly at
# different times depending on what other things the computer is running, we run
# the operation we wish to time several times and average the results:

# Below we have the time calculation to indicate speed of code execution between PyOpenCL and Python
M = 3
times = []
for i in xrange(M):
    start = time.time()
    for i in xrange(0, N):
		py_val += a[i]*np.cos((i+1)*x) + b[i]*np.sin((i+1)*x) 
    times.append(time.time()-start)
print 'python time:  ', np.average(times)

times = []
for i in xrange(M):
    start = time.time()
    prg.bandlimfunc(queue, x.shape, None, a_buf, b_buf, y_buf, x_buf, np.int32(N))
    times.append(time.time()-start)
print 'opencl time:  ', np.average(times)