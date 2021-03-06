#ifndef TOPOGRAPHIC_ANISOTROPY_LARGERGRID_H
#define TOPOGRAPHIC_ANISOTROPY_LARGERGRID_H

typedef struct
{
	//Host-side input data
	int NumRows;

	int NumCols;
	
	int *h_data;
	float *h_anisotropy,*h_azimuth,*h_angle;

	//Device buffers
	int *d_data;
	float *d_anisotropy,*d_azimuth,*d_angle;


	//Reduction copied back from GPU
	//float *h_Sum_from_device;

	//Stream for asynchronous command execution
	cudaStream_t stream;
	cudaStream_t stream2;

} GPU_struct;

#endif
