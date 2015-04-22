################################################################################################
Histogram Calculation for N=R*C bytes image where R and C are byte dimensions.
Pixels are 8 bit values from 0 - 255

By Avinash Sridhar (avinash.sridhar1989@gmail.com)
################################################################################################
READ ME FILE FOR algo-histogram.py
###################################################################################
PLATFORM: NVIDIA CUDA
GPU: TESLA K-40 C

Execute the code on GPU using normal procedure.

###################################################################################
BRIEF EXPLANATION ON ALGORITHM USED###################################################################################

Basic Algorithm: This is the algorithm provided by the TA. This shows a basic approach of histogram calculation on the GPU.

My Algorithm: We perform the histogram computation through usage of local memory. Basically, we perform a map of each thread (local Id) within a work group to
the histogram of a specific 8 bit pixel. Thus, each work group will have 256 work items and the total number of work groups = N / 256, where N is the total size of image.
During initial phase of algorithm, we initialize the work item in each work group to 0. Note that the barrier synchronization is performed to ensure that we proceed to 
while loop only when initialization to zero is completed in the work item of each work group. (Snippet below):
    __local unsigned int temp[256]
    temp[localId] = 0;
    barrier(CLK_LOCAL_MEM_FENCE)
	
In the next step, we iterate through the entire size of image by incrementing through an offset value. The offset value is given by: get_local_size(0)*get_num_groups(0) .
We put a while loop to ensure that we do not exceed the image’s total size. A barrier synchronization is used to ensure that we proceed to next stage only after processing 
all the writes on the thread.

Atomic increments are being used to prevent race conditions during increment of threads.
In the second phase of algorithm, we will increment the global memory variable final_bin by incrementing it with the value of local memory variable temp. 
Since temp has a size of 256 bytes, we can safely map each thread to the global memory variable. 


###################################################################################
