###################################################################################

Matrix Multiplication with Complex Numbers using PyOpenCL.
By Avinash Sridhar (as4626@columbia.edu)

###################################################################################

READ ME FILE FOR complex_matmul.py

###################################################################################

Execute the code on GPU using normal procedure.


If you have multiple jobs running on GPU, use slurm scheduler as follows:

sbatch --gres=gpu:1 --wrap="/opt/PYTHON/bin/python tiling_matmul.py"

###################################################################################
Explanation of two algorithms used:
#######################################################################################################

Algorithm – 1: We are multiplying the matrices by passing one global ID (number of rows is the global
size). To optimize the speed of multiplication, we are storing A’s rows within each item by storing it in
private memory. Also, we are making use of local memory by storing B’s columns in local / shared
memory.

#######################################################################################################

Algorithm – 2: Here, we are using the concept of tiling. In this algorithm we are using local_id(0) and
local_id(1). Within the kernel, we initiate two variables in the local memory i.e.
Ai[TILE_WIDTH][TILE_WIDTH] and Bi[TILE_WIDTH][TILE_WIDTH]. TILE_WIDTH refers to the width of
each tile (as asked in question, MATRIX DIMENSIONS will be a multiple of TILE_WIDTH). Ai and Bi will
store the values of A and B respectively in tile format to ensure that the accesses to global memory is
reduced. Within the kernel, we will store the values of Ai and Bi (local memory) by iterating through
each tile in A and B matrices. To ensure that we are synchronized within each block, we will use
BARRIER SYNCHRONIZATION.

Once the values are copied into the local memory, we will iterate through the TILE_WIDTH using
another for loop to store the multiplication product of tile matrices Ai and Bi into private memory
variable.

We will insert another barrier synchronization at the end of the previous for loop, to ensure that we
have the complete product stored into private memory from local memory before proceeding to the
next tile.

######################################################################################################
