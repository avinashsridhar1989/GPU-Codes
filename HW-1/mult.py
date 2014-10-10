import pyopencl as cl
import numpy as np
import os

ORDER = 2
LEN = ORDER*ORDER
ctx = cl.create_some_context()

commandQueue = cl.CommandQueue( ctx )

#A = np.array((72, 45, 75, 61), dtype = np.int32)
#B = np.array((26, 53, 46, 76), dtype = np.int32)
d = np.array(([1,1,3,12],[0,13,1,12],[13,14,15,16]), dtype = np.int32)
e = np.array(([0,0,12,12,1],[0,13,0,13,7],[1,2,3,2,12],[1,4,56,2,1]), dtype = np.int32)
C = np.empty_like(A)

in_buf1 = cl.Buffer( ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                 hostbuf = A )
in_buf2 = cl.Buffer( ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                 hostbuf = B )
out_buf = cl.Buffer( ctx, cl.mem_flags.WRITE_ONLY, C.nbytes )

kernelSrc1 = """__kernel void
            matrixMul(  /*const int Mdim,
                        const int Ndim,
                        const int Pdim,*/
                        __global int* C,
                        __global int* A,
                        __global int* B,
                        int wA, int wB)
           {
                int row = get_global_id(1); //2D Threas ID x
                int col = get_global_id(0); //2D Threas ID y                

                //Perform dot-product accumulated into value
                int value = 0;
                for ( int k = 0; k < wA; k++ ){
                    value += A[row*wA + k] * B[k*wB+col];
                }
                C[row*wA+col] = value; //Write to the device memory
            }"""

program1 = cl.Program(ctx, kernelSrc1 ).build()
event1 = program1.matrixMul( commandQueue, (3,5), None,
                     out_buf, in_buf1, in_buf2, np.int32(4), np.int32(5));
event1.wait()

cl.enqueue_copy(commandQueue, C, out_buf)
print C