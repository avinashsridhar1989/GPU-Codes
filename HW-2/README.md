###################################################################################
Matrix Multiplication with Complex Numbers using PyOpenCL.
By Avinash Sridhar (avinash.sridhar1989@gmail.com)
###################################################################################
READ ME FILE FOR complex_matmul.py
###################################################################################

Execute the code on GPU using normal procedure.

If you have multiple jobs running on GPU, use slurm scheduler as follows:

sbatch --gres=gpu:1 --wrap="/opt/PYTHON/bin/python complex_matmul.py"
###################################################################################
TO PLOT GRAPHS
###################################################################################
If in case you would like to see graphs plotted for each of the four algorithms, please set plot_value = True.
This will ensure that True is passed to 'def plotMaker' function helping us to plot four sets of graphs for the
four algorithms.

###################################################################################
BRIEF EXPLANATION ON 4 ALGORITHMS
###################################################################################

Algorithm-1: Here we are multiplying the two matrices by passing two global ids ([L,N] will be the global size) ie.
get_global_id(0) and get_global_id(1). The number of work items is L x N (final matrix C).

Algorithm-2: We are multiplying the matrices by passing one global ID (number of rows is the global size). 
To optimize the speed of multiplication, we are iterating through C’s row of items (M) in each work item.

Algorithm-3: We are multiplying the matrices by passing one global ID (number of rows). To optimize the speed of 
multiplication, we are iterating through C’s row of items (M) in each work item and also storing A’s rows 
within each item by storing it in private memory.

Algorithm-4: We are multiplying the matrices by passing one global ID (number of rows is the global size). 
To optimize the speed of multiplication, we are storing A’s rows within each item by storing it in private memory. 
Also, we are making use of local memory by storing B’s columns in local / shared memory.
###################################################################################
