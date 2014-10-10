#!/usr/bin/env python

"""
Matrix Multiplication with Complex Numbers using PyOpenCL.
By Avinash Sridhar (as4626@columbia.edu)

Note: Open README.md to see brief details of algorithms
"""

import time
import pyopencl as cl
import numpy as np
import numpy.matlib

plot_value = True
# Here we are defining two matrices a and b of size LxM and MxN for test purposes
L = 40
M = 60
N = 100
max_range = 10

# Function for plotting the graphs for each of the four algorithms
def plotMaker(MAKE_PLOT) :
	import matplotlib as mpl
	mpl.use('agg')
	import matplotlib.pyplot as plt
	if MAKE_PLOT == True :

# We plot for algorithm-1, keeping N constant and varying L,M in ratio of L:M = 1:2
		elements = []
		times = []
		prg = cl.Program(ctx, kernel_1).build()
		L = 1
		M = 2
		N = 200
		while L < N and M < N :
			elements.append(L*M)
			start = time.time()
			prg.matmul1(queue, (L, N), None, a_buf, b_buf, c1_buf, np.int32(L), np.int32(M), np.int32(N))
			times.append(time.time() - start)
			L = L + 1
			M = L*2
		plt.figure(1)
		plt.gcf()
		plt.plot(elements, times, 'ro-')
		plt.axis([0,10000,0.00008, 0.00015])
		plt.xlabel('Input Elements for Algorithm-1')
		plt.ylabel('Execution Times')
		plt.savefig('plot-algo-1-temp.png')

# We plot for algorithm-1, keeping L constant and varying M,N in ratio of M:N = 1:2
		elements = []
		times = []
		L = 200
		M = 1
		N = 2
		while M < L and N < L :
			elements.append(M*N)
			start = time.time()
			prg.matmul1(queue, (L, N), None, a_buf, b_buf, c1_buf, np.int32(L), np.int32(M), np.int32(N))
			times.append(time.time() - start)
			M = M + 1
			N = M*2
		plt.gcf()
		plt.plot(elements, times, 'bo-')
		plt.axis([0,10000,0.00008, 0.00015])
		plt.xlabel('Input Elements for Algorithm-1')
		plt.ylabel('Execution Times')
		plt.savefig('plot-algo-1.png')

# We plot for algorithm-2, keeping N constant and varying L,M in ratio of L:M = 1:2
		elements = []
		times = []
		prg = cl.Program(ctx, kernel_2).build()
		L = 1
		M = 2
		N = 200
		while L < N and M < N :
			elements.append(L*M)
			start = time.time()
			prg.matmul2(queue, (L, ), None, a_buf, b_buf, c2_buf, np.int32(L), np.int32(M), np.int32(N))
			times.append(time.time() - start)
			L = L + 1
			M = L*2
		plt.figure(2)
		plt.gcf()
		plt.plot(elements, times, 'ro-')
		plt.axis([0,10000,0.00008, 0.00015])
		plt.xlabel('Input Elements for Algorithm-2')
		plt.ylabel('Execution Times')
		plt.savefig('plot-algo-2-temp.png')

# We plot for algorithm-2, keeping L constant and varying M,N in ratio of M:N = 1:2
		elements = []
		times = []
		L = 200
		M = 1
		N = 2
		while M < L and N < L :
			elements.append(M*N)
			start = time.time()
			prg.matmul2(queue, (L, ), None, a_buf, b_buf, c2_buf, np.int32(L), np.int32(M), np.int32(N))
			times.append(time.time() - start)
			M = M + 1
			N = M*2
		plt.gcf()
		plt.plot(elements, times, 'bo-')
		plt.axis([0,10000,0.00008, 0.00015])
		plt.xlabel('Input Elements for Algorithm-2')
		plt.ylabel('Execution Times')
		plt.savefig('plot-algo-2.png')

# We plot for algorithm-3, keeping N constant and varying L,M in ratio of L:M = 1:2
		elements = []
		times = []
		prg = cl.Program(ctx, kernel_3).build()
		L = 1
		M = 2
		N = 200
		while L < N and M < N :
			elements.append(L*M)
			start = time.time()
			prg.matmul3(queue, (L, ), None, a_buf, b_buf, c3_buf, np.int32(L), np.int32(M), np.int32(N))
			times.append(time.time() - start)
			L = L + 1
			M = L*2
		plt.figure(3)
		plt.gcf()
		plt.plot(elements, times, 'ro-')
		plt.axis([0,10000,0.00008, 0.00015])
		plt.xlabel('Input Elements for Algorithm-3')
		plt.ylabel('Execution Times')
		plt.savefig('plot-algo-3-temp.png')

# We plot for algorithm-3, keeping L constant and varying M,N in ratio of M:N = 1:2
		elements = []
		times = []
		L = 200
		M = 1
		N = 2
		while M < L and N < L :
			elements.append(M*N)
			start = time.time()
			prg.matmul3(queue, (L, ), None, a_buf, b_buf, c3_buf, np.int32(L), np.int32(M), np.int32(N))
			times.append(time.time() - start)
			M = M + 1
			N = M*2
		plt.gcf()
		plt.plot(elements, times, 'bo-')
		plt.axis([0,10000,0.00008, 0.00015])
		plt.xlabel('Input Elements for Algorithm-3')
		plt.ylabel('Execution Times')
		plt.savefig('plot-algo-3.png')

# We plot for algorithm-4, keeping N constant and varying L,M in ratio of L:M = 1:2
		elements = []
		times = []
		prg = cl.Program(ctx, kernel_4).build()
		L = 1
		M = 2
		N = 200
		while L < N and M < N :
			elements.append(L*M)
			start = time.time()
			prg.matmul4(queue, (L, ), (1, ), a_buf, b_buf, c4_buf, np.int32(L), np.int32(M), np.int32(N))
			times.append(time.time() - start)
			L = L + 1
			M = L*2
		plt.figure(4)
		plt.gcf()
		plt.plot(elements, times, 'ro-')
		plt.axis([0,10000,0.00007, 0.0001])
		plt.xlabel('Input Elements for Algorithm-4')
		plt.ylabel('Execution Times')
		plt.savefig('plot-algo-4-temp.png')

# We plot for algorithm-4, keeping L constant and varying M,N in ratio of M:N = 1:2
		elements = []
		times = []
		L = 2
		M = 3
		N = 6
		while M < 200 and N < 200 :
			elements.append(M*N)
			start = time.time()
			prg.matmul4(queue, (L, ), (1, ), a_buf, b_buf, c4_buf, np.int32(L), np.int32(M), np.int32(N))
			times.append(time.time() - start)
			M = M + 1
			N = M*2
		plt.gcf()
		plt.plot(elements, times, 'bo-')
		plt.axis([0,10000,0.00007, 0.0001])
		plt.xlabel('Input Elements for Algorithm-4')
		plt.ylabel('Execution Times')
		plt.savefig('plot-algo-4.png')
	print "\nGraphs plotted! \nOpen repective algorithm graphs:\nplot-algo-1.png \nplot-algo-2.png \nplot-algo-3.png \nplot-algo-4.png \n"

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

# Matrix Multiplication Algorithm - 1, two global IDs
kernel_1 = """
#include <pyopencl-complex.h>
__kernel void matmul1(__global cfloat_t* A, __global cfloat_t* B, __global cfloat_t* C , const int L, const int M, const int N) {
	
	unsigned int row = get_global_id(0);
	unsigned int col = get_global_id(1);
	cfloat_t temp_val = 0;
	for (int k = 0; k < M; k++)
	{
		temp_val = temp_val + cfloat_mul(A[row*M + k], B[k*N+col]);
	}
	C[row*N + col] = temp_val;
}
"""

# Matrix Multiplication Algorithm - 2, single global ID, and we work one row at a time instead of per item
kernel_2 = """
#include <pyopencl-complex.h>
__kernel void matmul2(__global cfloat_t* A, __global cfloat_t* B, __global cfloat_t* C , const int L, const int M, const int N) {
	
	unsigned int col, k;
	unsigned int row = get_global_id(0);
	for (col = 0; col < N ; col++)
	{
		cfloat_t temp_val = 0;
		for (k=0; k < M ; k++)
		{
			temp_val = temp_val + cfloat_mul(A[row*M + k], B[k*N + col]);
		}
		C[row*N + col] = temp_val;
	}
}
"""
# Matrix Multiplication Algorithm - 3, we are not trying to use one row matrix in the private memory. Speeding Computation.
kernel_3 = """
#include <pyopencl-complex.h>
__kernel void matmul3(__global cfloat_t* A, __global cfloat_t* B, __global cfloat_t* C , const int L, const int M, const int N) {
	
	unsigned int col, k;
	cfloat_t A_privateMem[1024];
	unsigned int row = get_global_id(0);
	for (k = 0; k < M; k++)
	{
		A_privateMem[k] = A[row*M + k];
	}
	for (col = 0; col < N ; col++)
	{
		cfloat_t temp_val = 0;
		for (k=0; k < M ; k++)
		{
			temp_val = temp_val + cfloat_mul(A_privateMem[k], B[k*N + col]);
		}
		C[row*N + col] = temp_val;
	}
}
"""
# Matrix Multiplication Algorithm - 4, we are not trying to use one row matrix A into private memory and
# column matrix B into Local Memory. Speeding computation further.
kernel_4 = """
#include <pyopencl-complex.h>
__kernel void matmul4(__global cfloat_t* A, __global cfloat_t* B, __global cfloat_t* C, const int L, const int M, const int N) {
	
	__local cfloat_t B_localMem[1024];
	cfloat_t A_privateMem[1024];
	unsigned int row = get_global_id(0);
	unsigned int col_Loc = get_local_id(0);
	unsigned int nCol = get_local_size(0);
	unsigned int col, k;
	for (int k = 0; k < M; k++)
	{
		A_privateMem[k] = A[row*M + k];
	}
	for (int col = 0; col < N ; col++)
	{
		for (int k = col_Loc; k < N; k += nCol)
		{
			B_localMem[k] = B[k*N + col];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		cfloat_t temp_val = 0;
		for (int k=0; k < M ; k++)
		{
			temp_val = temp_val + cfloat_mul(A_privateMem[k], B_localMem[k]);
		}
		C[row*N + col] = temp_val;
	}
}
"""
# Generate random complex numbers for matrix values
a = np.random.randint(0, max_range, size = (L,M)).astype(np.complex64) + 1j*np.random.randint(0, max_range, size = (L,M)).astype(np.complex64)
b = np.random.randint(0, max_range, size = (M,N)).astype(np.complex64) + 1j*np.random.randint(0, max_range, size = (M,N)).astype(np.complex64)

print a
print b

# py_val is equivalent to y here, except that py_val will do the matrix multiplication in numerical python
py_val = np.dot(a,b)
c1 = np.zeros_like(py_val)
c2 = np.zeros_like(py_val)
c3 = np.zeros_like(py_val)
c4 = np.zeros_like(py_val)

# You need to set the flags of the buffers you create properly; otherwise,
# you might not be able to read or write them as needed:
mf = cl.mem_flags

# Create buffers
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
c1_buf =cl.Buffer(ctx, mf.WRITE_ONLY, c1.nbytes)
c2_buf =cl.Buffer(ctx, mf.WRITE_ONLY, c2.nbytes)
c3_buf =cl.Buffer(ctx, mf.WRITE_ONLY, c3.nbytes)
c4_buf =cl.Buffer(ctx, mf.WRITE_ONLY, c4.nbytes)

# Launch Kernel
# Build kernel for matrix multiplication method 1
prg = cl.Program(ctx, kernel_1).build()

# For LXM and MXN matrices, pass (L,N) as global id values and widths for L, M, N considering LxM, MxN..
prg.matmul1(queue, (L, N), None, a_buf, b_buf, c1_buf, np.int32(L), np.int32(M), np.int32(N))

# Retrieve the results from the GPU for matmul1:
cl.enqueue_copy(queue, c1, c1_buf)

# Build kernel for matrix multiplication method 2
prg = cl.Program(ctx, kernel_2).build()
prg.matmul2(queue, (L, ), None, a_buf, b_buf, c2_buf, np.int32(L), np.int32(M), np.int32(N))

# Retrieve the results from the GPU for matmul2:
cl.enqueue_copy(queue, c2, c2_buf)

# Build kernel for matrix multiplication method 3
prg = cl.Program(ctx, kernel_3).build()
prg.matmul3(queue, (L, ), None, a_buf, b_buf, c3_buf, np.int32(L), np.int32(M), np.int32(N))

# Retrieve the results from the GPU for matmul3:
cl.enqueue_copy(queue, c3, c3_buf)

# Build kernel for matrix multiplication method 4
prg = cl.Program(ctx, kernel_4).build()
prg.matmul4(queue, (L, ), (1, ), a_buf, b_buf, c4_buf, np.int32(L), np.int32(M), np.int32(N))

# Retrieve the results from the GPU for matmul3:
cl.enqueue_copy(queue, c4, c4_buf)

# Print values of all calculated matrices and compare
print '\nNumpy Matrix Multiplication :  ', py_val
print '\nOpencl Matrix Multiplication Algorithm 1 : ', c1
print '\nOpencl Matrix Multiplication Algorithm 2 :', c2
print '\nOpencl Matrix Multiplication Algorithm 3 :', c3
print '\nOpencl Matrix Multiplication Algorithm 4 :', c4

# Print the data type used for calculations
print '\nData Type used: ', py_val.dtype, '\n'

# Compare the calculated values for algorithms and numpy
print 'PyopenCL matrix multiply algorithm 1 and numpy are equal:        ', np.allclose(py_val, c1, rtol = 1e-02)
print 'PyopenCL matrix multiply algorithm 2 and numpy are equal:        ', np.allclose(py_val, c2, rtol = 1e-02)
print 'PyopenCL matrix multiply algorithm 3 and numpy are equal:        ', np.allclose(py_val, c3, rtol = 1e-02)
print 'PyopenCL matrix multiply algorithm 4 and numpy are equal:        ', np.allclose(py_val, c4, rtol = 1e-02)

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
	prg.matmul1(queue, (L, N), None, a_buf, b_buf, c1_buf, np.int32(L), np.int32(M), np.int32(N))
	times.append(time.time()-start)
times1 = np.average(times)
print 'OpenCL Algorithm-1 time:  ', times1

# Measure time taken by Algorithm 2
prg = cl.Program(ctx, kernel_2).build()
times = []
for i in xrange(M):
	start = time.time()
	prg.matmul2(queue, (L, ), None, a_buf, b_buf, c2_buf, np.int32(L), np.int32(M), np.int32(N))
	times.append(time.time()-start)
times2 = np.average(times)
print 'OpenCL Algorithm-2 time:  ', times2

# Measure time taken by Algorithm 3
prg = cl.Program(ctx, kernel_3).build()
times = []
for i in xrange(M):
	start = time.time()
	prg.matmul3(queue, (L, ), None, a_buf, b_buf, c3_buf, np.int32(L), np.int32(M), np.int32(N))
	times.append(time.time()-start)
times3 = np.average(times)
print 'OpenCL Algorithm-3 time:  ', times3

# Measure time taken by Algorithm 4
prg = cl.Program(ctx, kernel_4).build()
times = []
for i in xrange(M):
	start = time.time()
	prg.matmul4(queue, (L, ), (1,), a_buf, b_buf, c4_buf, np.int32(L), np.int32(M), np.int32(N))
	times.append(time.time()-start)
times4 = np.average(times)
print 'OpenCL Algorithm-4 time:  ', times4

# Execute plots, ** this is dependent on plot_value set to True/False at top**
plotMaker(plot_value)
