###################################################################################
Bandwidth Limited Function using PyOpenCL.
By Avinash Sridhar (avinash.sridhar1989@gmail.com)
###################################################################################
READ ME FILE FOR gen_bw_final.py
###################################################################################

Execute the code on GPU using normal procedure.

If you have multiple jobs running on GPU, use slurm scheduler as follows:

sbatch --gres=gpu:1 --wrap="/opt/PYTHON/bin/python gen_bw_final.py"

###################################################################################
BRIEF EXPLANATION ON CODE
###################################################################################

“gen_bw_final.py” performs addition of waves of different frequencies and different amplitudes. Each step in the i iteration calculate a wave of frequency f = (i + 1)/2π, and a[i], b[i] are the amplitude generated randomly before.

The PyOpenCL code is provided as a Python file named gen_bw_final.py

The code is provided for N = 4 frequency components and Interval = 120 ( dx = 0.05 and x_max = 6.0 ).

Vary the frequency components and Interval to observe results. You will notice significant difference with GPU execution time
once you hit a high number of frequency components and interval number.

###################################################################################
