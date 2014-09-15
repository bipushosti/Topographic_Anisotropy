

import numpy as np
from numba import *
from numbapro import cuda



#@cuda.jit('void(float32[:],float32[:],float32[:])')
@cuda.jit('void(float32[:],float32[:],float32[:])')
def square_add(a,b,c):
	tx = cuda.threadIdx.x
	bx = cuda.blockIdx.x
	bw = cuda.blockDim.x

	i = tx + bx * bw

	if (i>a.shape[0]):
		return
	else:
		c[i] = a[i]*a[i] + b[i]*b[i]

	
	cuda.syncthreads()



#@cuda.jit('void(float32[:],float32[:])')
@cuda.jit('void(float32,float32[:],float32[:])')
def twice(val,a,temp):
	i = cuda.grid(1)
	
	if (i<a.shape[0]):
		temp[i] = a[i] * val
	

#a = np.array([],dtype = np.float32)
a = np.array(range(0,1024),dtype = np.float32)
#b = np.array([],dtype = np.float32)
#b = np.array(range(41724,83448),dtype=np.float32)

#c = np.array([],dtype = np.float32)
#c = np.zeros(shape=(1,41724),dtype=np.float32)
val = 5
temp = np.empty(1024,dtype=np.float32)
#temp  = np.zeros(shape=(1,1024),dtype=np.float32)
d_a = cuda.to_device(a)
d_val = cuda.to_device(val)
#d_b = cuda.to_device(b)
#d_c = cuda.to_device(c,copy=False)
d_temp = cuda.to_device(temp,copy=False)
#square_add[(41,1),(1024,1)](d_a,d_b,d_c)
#square_add[(1,82),(1,512)](d_a,d_b,d_c)
twice[(2,1),(512,1)](d_val,d_a,d_temp)
temp = d_temp.copy_to_host()
print temp
#c = d_c.copy_to_host()
#print temp
#print len(temp[0])
