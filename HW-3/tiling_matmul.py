#!/usr/bin/env python

"""
Matrix Calculation for A*(B+C) where A,B,C are square matrices using Real Numbers using PyOpenCL.
By Avinash Sridhar (as4626@columbia.edu)

Note: Open README.md to see brief details of algorithms
"""

import time
import pyopencl as cl
import numpy as np
import numpy.matlib

# Here we are defining the sizes of LxMxN. For HW-3, since we are calculating A*(B+C), we will keep L=M=N= as a multiple of block_size
# where block_size is TIILE_WIDTH (in kernel)
L = 1000
M = 1000
N = 1000
block_size = 25
workgroup_items = 25
max_range = 10

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

# Matrix Multiplication Algorithm - 1, we are trying to use one row matrix A into private memory and
# column matrix B into Local Memory.
kernel_1 = """
__kernel void matmul1(__global float* A, __global float* B, __global float* C, __global float* output, const int L, const int M, const int N) {
	
	__local float B_localMem[1024];
	float A_privateMem[1024]; // We define private memory
	unsigned int row = get_global_id(0); // This will get global dimensions from Python, i.e. L
	unsigned int col_Loc = get_local_id(0); // Number of items per work group is defined in python as workgroup_items
	unsigned int nCol = get_local_size(0); // The local size will be L/workgroup_items
	unsigned int col, k;
	
	// Calculate an entire row and store in private memory
	for (int k = 0; k < M; k++)
	{
		A_privateMem[k] = A[row*M + k];
	}
	// Here we iterate through all the columns, i.e. N
	for (int col = 0; col < N ; col++)
	{	
		/* Here we will store an entire column of matrix data into local memory. The local memory variable is B_localMem[]. We will
		also perform Barrier Synchronization to ensure that we go to next stage of loop only once all items of the local memory are
		calculated */
		
		for (int k = col_Loc; k < M; k += nCol)
		{
			B_localMem[k] = B[k*N + col] + C[k*N + col];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		float temp_val = 0;
		for (int k=0; k < M ; k++)
		{
			temp_val = temp_val + A_privateMem[k]*B_localMem[k];
		}
		output[row*N + col] = temp_val;
	}
}
"""

kernel_2 = """
#define TILE_WIDTH 25
__kernel void matmul2(__global float* A, __global float* B, __global float* C, __global float* output, const int L, const int M, const int N) {
	
		unsigned int tx = get_local_id(0); // Obtain local id (0) which is number of items per block_size
		unsigned int ty = get_local_id(1); // Obtain local id (1) which is again number of items per block_size
		unsigned int bx = get_group_id(0); // This is analagous to block id in CUDA, i.e. we obtain block id (0)
		unsigned int by = get_group_id(1); // This is analogous to block id in CUDA, i.e. we obtain block id (1)

		/* We define Ai[][] and Bi[][], which are stored in local memory. These variables will store the A[][] and B[][] values of each
		tile to ensure that number of global memory accesses is reduced to speed computation */

		__local float Ai[TILE_WIDTH][TILE_WIDTH];
		__local float Bi[TILE_WIDTH][TILE_WIDTH];
		
		/* row and column are actually the same as global id (0) and global id(1) */

		unsigned int row = by*get_local_size(1) + ty;
		unsigned int col = bx*get_local_size(0) + tx;
		float CL_privateMem = 0;
		for (int k = 0; k < M/TILE_WIDTH; k++)
		{
			Ai[ty][tx] = A[row*M + k*TILE_WIDTH + tx];
			Bi[ty][tx] = B[(k*TILE_WIDTH + ty)*N + col] + C[(k*TILE_WIDTH + ty)*N + col];

			/* Here we insert a barrier to ensure that we will not proceed with performing the multiplication within the tile. only
			once we have all the values within a tile from the local memory, will we proceed to next step */

			barrier(CLK_LOCAL_MEM_FENCE);

			/* Store the multiplied results from tile locations into private memory */

			for (int j = 0; j < TILE_WIDTH; j++)
			{
				CL_privateMem = CL_privateMem + Ai[ty][j]*Bi[j][tx];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		output[row*N + col] = CL_privateMem;
}
"""

# Generate random complex numbers for matrix values
a = np.random.randint(0, max_range, size = (L,M)).astype(np.float32)
b = np.random.randint(0, max_range, size = (M,N)).astype(np.float32)
c = np.random.randint(0, max_range, size = (M,N)).astype(np.float32)
print a
print b
print c

# py_val is equivalent to y here, except that py_val will do the matrix multiplication in numerical python
py_val = np.dot(a,(b+c))
d = b + c

# These will store the outputs for algorithms 1 and 2
out1 = np.zeros_like(py_val)
out2 = np.zeros_like(py_val)

# You need to set the flags of the buffers you create property; otherwise,
# you might not be able to read or write them as needed:
mf = cl.mem_flags

# Create buffers
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
c_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
out1_buf =cl.Buffer(ctx, mf.WRITE_ONLY, out1.nbytes)
out2_buf =cl.Buffer(ctx, mf.WRITE_ONLY, out2.nbytes)

# Build kernel for matrix multiplication method 1
prg = cl.Program(ctx, kernel_1).build()
prg.matmul1(queue, (L, ), (L/workgroup_items, ), a_buf, b_buf, c_buf, out1_buf, np.int32(L), np.int32(M), np.int32(N))

# Retrieve the results from the GPU for matmul1:
cl.enqueue_copy(queue, out1, out1_buf)

# Build kernel for matrix multiplication method 
prg = cl.Program(ctx, kernel_2).build()
prg.matmul2(queue, (L,N), (block_size,block_size), a_buf, b_buf, c_buf, out2_buf, np.int32(L), np.int32(M), np.int32(N))

# Retrieve the results from the GPU for matmul2:
cl.enqueue_copy(queue, out2, out2_buf)

# Print values of all calculated matrices and compare
print '\nNumpy Matrix Multiplication :  ', py_val
print '\nOpencl Matrix Multiplication Algorithm 1 :', out1
print '\nOpencl Matrix Multiplication Algorithm 2 :', out2

# Print the data type used for calculations
print '\nData Type used: ', py_val.dtype, '\n'

# Compare the calculated values for algorithms and numpy
print 'PyopenCL matrix multiply algorithm 1 and numpy are equal:        ', np.allclose(py_val, out1)
print 'PyopenCL matrix multiply algorithm 2 and numpy are equal:        ', np.allclose(py_val, out2)

# Below we have the time calculation to indicate speed of code execution between PyOpenCL algorithms and Python
M = 3

# Measure time taken by Python Numpy
times = []
for i in xrange(M):
	start = time.time()
	py_val = np.dot(a,b)
	times.append(time.time()-start)
times_py = np.average(times)
print 'python time:  ', times_py

# Measure time taken by Algorithm 1
prg = cl.Program(ctx, kernel_1).build()
times = []
for i in xrange(M):
	start = time.time()
	prg.matmul1(queue, (L, ), (L/workgroup_items, ), a_buf, b_buf, c_buf, out1_buf, np.int32(L), np.int32(M), np.int32(N))
	times.append(time.time()-start)
times1 = np.average(times)
print 'OpenCL Algorithm-1 time:  ', times1

# Measure time taken by Algorithm 1
prg = cl.Program(ctx, kernel_2).build()
times = []
for i in xrange(M):
	start = time.time()
	prg.matmul2(queue, (L,N), (block_size,block_size), a_buf, b_buf, c_buf, out2_buf, np.int32(L), np.int32(M), np.int32(N))
	times.append(time.time()-start)
times2 = np.average(times)
print 'OpenCL Algorithm-2 time:  ', times2
