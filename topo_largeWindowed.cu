//Input file: space delimited

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <limits.h>

#include <unistd.h>
#include <ctype.h>
#include <getopt.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime.h>
//#include <helper_functions.h>
//#include <helper_cuda.h>
#include <assert.h>

#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

#include "topoMultiGPU_Header.h"

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

//Size of the GPU memory
#define GPU_MEMSIZE_GB		6

//For case in which XSIZE = 1201 and YSIZE = 801
#define GLOBAL_MEM_USE_MB	773
#define MEM_USE_PER_THREAD_B	1280

//MAX_XSIZE_POSSIBLE is the maximum size of x or max number of columns if there is only one row
#define MAX_XSIZE_POSSIBLE	7483647
//#define MAX_XSIZE_POSSIBLE	floor(((GPU_MEMSIZE_GB * 1000 - GLOBAL_MEM_USE_MB)*1000000)/MEM_USE_PER_THREAD_B) 


//#define XSIZE 		1201
//#define YSIZE			801


//Always have even number of radius;and divisible by 10
//#define RADIUS			100

#define	RADSTEP			1

using namespace std;

//---------------------------- Grid Layout ------------------------------------------------------------------------------------------------------//
/*

	|
	|
	|
Angles	|
	|
      	|
	------------------------------------------
		Radius (Divided by radiusDiv)

	Max the angles can be is 36;

*/	


//Changes
//March 6: 
//		Individual if(threadIdx.x < (dividedRadius * angleSize)) is replaced by one long condition (Tested + Works)
//		

//---------------------------Function and Global variable declarations--------------------------------------------------------------------------//
__global__ void getMatrix(const int* __restrict__ data,const float* __restrict__ angle,float* __restrict__ anisotropy,float* __restrict__ azimuth,long int XSIZE,long int YSIZE,int RADIUS,int angleSize,int radiusDiv);
__device__ float check_if_zero(float value_to_check,float functionArg1, float functionArg2);

//__global__ void getMatrix(int* data,float* angle,float* anisotropy,float* azimuth,long int XSIZE,long int YSIZE,int RADIUS,int angleSize,int radiusDiv);
int Get_GPU_devices();
static void HandleError( cudaError_t err,const char *file, int line);
inline cudaError_t checkCuda(cudaError_t result);

//-----------------------------------------------Device Functions-----------------------------------------------------------//

//Function that returns avg_valueOrtho if avg_value ==0 && avg_valueOrtho <1 && avg_valueOrtho >0
//Or returns 1 otherwise
__device__ float calculate_averageValue(float avg_value,float avg_valueOrtho)
{

	float x1,x2,x3;
	//float f1,f2,f3;

	x1 = ceilf(avg_value/FLT_MAX);
	x2 = ceilf(floorf(avg_valueOrtho)/FLT_MAX);
	x3 = ceilf(avg_valueOrtho/FLT_MAX);	
/*	
	f3 = x3 * avg_valueOrtho + (1 - x3);
	f2 = x2 + (1 - x2) * f3;
	f1 = x1 * avg_value + (1 - x1) * f2;
*/

	return (x1 * avg_value + (1 - x1) * (x2 + (1 - x2) * (x3 * avg_valueOrtho + (1 - x3))));

}

//Function that returns 1 if averageValueOrtho == 0 and averageValueOrtho otherwise
__device__ float calculate_averageValueOrtho(float averageValueOrtho)
{
	return (ceil(averageValueOrtho/FLT_MAX) * averageValueOrtho + (1 - ceil(averageValueOrtho/FLT_MAX)) * 1);

}

//Function that emulates if(a<b){...} but without the if statement to avoid warp divergence
__device__ float condition_if_lessthan(float valueA,float valueB,float result_ifAIsLessThanB, float result_ifAIsNotLessThanB)
{

	return ceilf(floorf(valueA/valueB)/FLT_MAX) * result_ifAIsNotLessThanB + (1.0 -  ceilf(floorf(valueA/valueB)/FLT_MAX) ) * result_ifAIsLessThanB;


}


//--------------------------------------------------------------------------------------------------------------------------//

__global__ void getMatrix(const int* __restrict__ data,const float* __restrict__ angle,float* __restrict__ anisotropy,float* __restrict__ azimuth,long int XSIZE,long int YSIZE,int RADIUS,int angleSize,int radiusDiv)
//__global__ void getMatrix(int* data,float* angle,float* anisotropy,float* azimuth,long int XSIZE,long int YSIZE,int RADIUS,int angleSize,int radiusDiv)
{

	//Block Indices
	//int block_id =  blockIdx.y * gridDim.x + blockIdx.x;

	//Thread indices; Using only the x dimensions
	//int thread_id = block_id * blockDim.x + threadIdx.x;

	//The entire radius cannot be used to create shared memory; So a smaller radius size is used
	//When RADIUS = 100 and radiusDiv = 5, dividedRadius=20
	int dividedRadius = RADIUS/radiusDiv;

	//int thread_y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int thread_x = 0;
	int thread_y = 0;
	
	//Data indices
	int dataIdx_x = blockIdx.x + RADIUS;
	int dataIdx_y = blockIdx.y + RADIUS;
	


//----------------------------------------------------------------------------------------------------------------------------//	
	//Shared memory that holds avg_value and avg_valueOrtho
	extern __shared__ float averages[];

	//if(thread_y == 0 && thread_x == 0){printf("Inside the kernel\n");}


	int xrad,yrad,xradOrtho1,yradOrtho1,xradOneEighty,yradOneEighty,valueOneEighty;
	int xradOrtho2,yradOrtho2,i;

	int sum_value,sum_valueOrtho;

	//int valueOrtho1,valueOrtho2;
	//float value;
	float avg_value,avg_valueOrtho;
	float avg_valueSum;


	//For 1 GPU
	if(threadIdx.x < (dividedRadius * angleSize)) //For radius=100, total threads per block=736, checks if less than 720 (idx starts at 0)
	{
//Needs review; Need to add CUDA float to int and back functions?
		thread_x = threadIdx.x - (int)((float)threadIdx.x / (float)dividedRadius) * dividedRadius; 
		thread_y = threadIdx.x / dividedRadius;

		

		sum_value = 0;
		sum_valueOrtho = 0;

		//Each thread in x-axis represents radiusDiv numbers
		for(i=0;i<radiusDiv;i++){


			//------------------------------------------------------------------------------------------------------------//	
			//Computation for angle of interest
			xrad = (int)lrintf(cosf(angle[thread_y]) * (thread_x * radiusDiv + i + 1) + dataIdx_x);	
			yrad = (int)lrintf(sinf(angle[thread_y]) * (thread_x * radiusDiv + i + 1) + dataIdx_y);	

//Works till here; xrad and yrad for this and the working versions are the same **


			//value = data[dataIdx_y * XSIZE + dataIdx_x]  - data[yrad * XSIZE + xrad];
			//value = value * value;
		
			//One eighty angle computation
			xradOneEighty = (int)lrintf(cosf(angle[thread_y]+3.14159) * (thread_x * radiusDiv + i + 1) + dataIdx_x);	
			yradOneEighty = (int)lrintf(sinf(angle[thread_y]+3.14159) * (thread_x * radiusDiv + i + 1) + dataIdx_y);	
		
			valueOneEighty = data[dataIdx_y * XSIZE + dataIdx_x] - data[yradOneEighty * XSIZE + xradOneEighty];
			valueOneEighty = valueOneEighty * valueOneEighty;

			//------------------------------------------------------------------------------------------------------------//
			//Computation for values on angle orthogonal to angle of interest
			xradOrtho1 = (int)lrintf(cosf(angle[thread_y]+3.14159/2) * (thread_x * radiusDiv + i + 1) + dataIdx_x);	
			yradOrtho1 = (int)lrintf(sinf(angle[thread_y]+3.14159/2) * (thread_x * radiusDiv + i + 1) + dataIdx_y);	
		
			//One eighty ortho angle computation
			xradOrtho2 = (int)lrintf(cosf(angle[thread_y]+3.14159*3/2) * (thread_x * radiusDiv + i + 1) + dataIdx_x);	
			yradOrtho2 = (int)lrintf(sinf(angle[thread_y]+3.14159*3/2) * (thread_x * radiusDiv + i + 1) + dataIdx_y);	

			//valueOrtho1 = data[dataIdx_y * XSIZE + dataIdx_x]  - data[yradOrtho1 * XSIZE + xradOrtho1];
			//valueOrtho1 = valueOrtho1 * valueOrtho1;
			//valueOrtho2 = data[dataIdx_y * XSIZE + dataIdx_x]  - data[yradOrtho2 * XSIZE + xradOrtho2];
			//valueOrtho2 = valueOrtho2 * valueOrtho2;

//Both parts of the sum value are same in this and the working version



//All rads and one eighty values work!
			//-------------------------------Getting the sum values-------------------------------------------------------//
			//sum_value = value + valueOneEighty;
			sum_value = data[dataIdx_y * XSIZE + dataIdx_x]  - data[yrad * XSIZE + xrad];

			sum_value = sum_value * sum_value;

//Same sum_values too!

			sum_value += valueOneEighty;	
//Sum_value is correct	

		
			//sum_valueOrtho = valueOrtho1 + valueOrtho2;
			sum_valueOrtho = data[dataIdx_y * XSIZE + dataIdx_x]  - data[yradOrtho1 * XSIZE + xradOrtho1];
			sum_valueOrtho = sum_valueOrtho * sum_valueOrtho; //valueOrtho1
//valueOrtho1 Or sum_valueOrtho till is point is correct
						

			sum_valueOrtho += (data[dataIdx_y * XSIZE + dataIdx_x]  - data[yradOrtho2 * XSIZE + xradOrtho2])*(data[dataIdx_y * XSIZE + dataIdx_x]  - data[yradOrtho2 * XSIZE + xradOrtho2]);
			
//Correct till here; Problem with the averages?

			//-----Storing the sum values in the shared memory array-----------------------------------------------------//
			averages[thread_y * RADIUS + thread_x * radiusDiv + i] = sum_value;
//Correct sum_values going to the correct positions

			averages[RADIUS * angleSize + thread_y * RADIUS + thread_x * radiusDiv + i] = sum_valueOrtho;



			//averages[2 * RADIUS * angleSize + thread_y * RADIUS + thread_x* radiusDiv + i] = sum_value;

			//printf("X Y Value dataIdx_x dataIdx_y xrad yrad %d %d %f %d %d %f %f \n",thread_x,thread_y,sum_value,dataIdx_x,dataIdx_y,xrad,yrad);
			//printf("X Y EleValue %d %d %f\n",thread_x,thread_y,data[thread_y * RADIUS + thread_x]);
//			printf("thread_x thread_y xrad yrad %d %d %d %d\n",thread_x,thread_y,xrad,yrad);
	
		}
	
			
	//}
		__syncthreads();



	//Single thread averaging over the row or x dimension (Using 1 thread per row)---------------------------------------------------------------------------//
	//The block only contains threads in x dimension. Therefore, threadIdx.x is the thread index.	
	//if(threadIdx.x < (dividedRadius * angleSize)){

		//Only getting thr threads from 0 to angleSize (0 to 35; ANGLESIZE = 36 is hard coded)
		// Going through each Anglesize (y-direction) and getting the average of the row (x-direction)
		if(threadIdx.x < angleSize)
		{

			//--------Getting the avg_value and storing it in the shared mem array "averages"-------------------------------------//
			//Can't use an int to store a float
			//sum_value = 0;

			avg_valueSum = 0.0;

			//Loop from the start of the row to the end of the row which is RADIUS away from the start
			for(i = 0; i < RADIUS; i++){
				avg_valueSum += averages[threadIdx.x * RADIUS + i];	
				avg_value = avg_valueSum/(2*(i+1));
				averages[threadIdx.x * RADIUS + i] = avg_value;

		

			}

			//-------Getting the avg_valueOrtho and storing it in the shared mem array---------------------------------------------//	
			//Can't use int to store a float
			//sum_valueOrtho = 0;

			//Reusing variable
			//Using avg_valueSum to store sum of avg_valueOrtho
			avg_valueSum = 0.0;

			//Looping through the row
			for(i = 0; i < RADIUS; i++){
				avg_valueSum += averages[RADIUS * angleSize + threadIdx.x * RADIUS + i];	
				avg_valueOrtho = avg_valueSum/(2*i+1);
				averages[RADIUS * angleSize + threadIdx.x * RADIUS + i] = avg_valueOrtho;

			}
		}
			//Now the first matrix has the averaged values (avg_value);
			//And the second has the averaged Ortho values (avg_valueOrtho);	
		//}




		__syncthreads();
	
//Correct upto this point;
//Result from this is in float while that from the previous version is in ints. Therefore two results differ by less than an int (Ex: 374 vs 374.200012)
//Need to change the original code to make sure the results are stored in floats by using CUDA's conversion functions

//------------------>> Fixed till here; Ints were used to store floats










	//-Multi-thread averaging----------------------------------------------------------------------------------------------------//	
//*************
//NEEDS TO BE IMPLEMENTED ONCE THE SINGE THREAD AVERAGE PART IS COMPLETE AND ACCURATE
//*************



/*
	//Uses multiple threads to get the average

	if((thread_x > 0) && (threadIdx.x < (dividedRadius * angleSize)))
	{
		int offsetStart,offsetEnd;

		//Getting the avg_value and storing it in the shared mem array "averages"
		for(radIdx=0;radIdx<radiusDiv;radIdx++){

			sum_value = 0;
			
			offsetStart = thread_y * RADIUS + thread_x * radiusDiv + radIdx;
			offsetEnd = thread_y * RADIUS;
 
			for(i = offsetStart; i>= offsetEnd; i--)
			{ 
			//for(i=(thread_x * radiusDiv + radIdx); i>=0; i--){
				sum_value += averages[i];
			}

			avg_value = sum_value/(2*(thread_x * radiusDiv + radIdx + 1));
			averages[2 * RADIUS * angleSize + thread_y * RADIUS + thread_x * radiusDiv + radIdx] = avg_value;
	
		}
		//Now the last matrix has the averaged values (avg_value);
		//First has the individual sum values; 
		//And the second has the individual sum of Ortho values;

		__syncthreads();
	
		//------------------------------------------------------------------------------------------------------------//	
		//Getting the avg_valueOrtho and storing it in the shared mem array
		for(radIdx=0;radIdx<radiusDiv;radIdx++){
		
			sum_valueOrtho = 0;

			offsetStart = RADIUS * angleSize + thread_y * RADIUS + thread_x * radiusDiv + radIdx;
			offsetEnd = RADIUS * angleSize + thread_y * RADIUS;

			for(i = offsetStart; i>= offsetEnd; i++)
			//for(i=(thread_x * radiusDiv + radIdx); i>=0; i--)
			{
				sum_valueOrtho += averages[i];
			}

			avg_valueOrtho = sum_valueOrtho/(2*(thread_x * radiusDiv + radIdx + 1));
			averages[thread_y * RADIUS + thread_x * radiusDiv + radIdx] = avg_valueOrtho;

		}

		//Now the first matrix has averaged Ortho values (avg_valueOrtho); 
		//The last matrix still has the averaged values (avg_values);
		//And the second has the individual sum of Ortho values;

		__syncthreads();
	}
*/
	//if(blockIdx.x==0 && blockIdx.y ==0){ printf("threadIdx.x = %d\n",threadIdx.x);}



	//Error Checking-------------------------------------------------------------------------------------------------------------//	
	//if(threadIdx.x < (dividedRadius * angleSize))
	//{
		//thread_x = threadIdx.x - (int)((float)threadIdx.x / (float)dividedRadius) * dividedRadius; 
		//thread_y = threadIdx.x / dividedRadius;
		//----Error checking-----------------------------------------------------------------------------------------//
		for(i=0;i<radiusDiv;i++){
			//Getting averages and storing them in variables 

			avg_value = averages[thread_y * RADIUS + thread_x * radiusDiv + i];			
			avg_valueOrtho = averages[ RADIUS * angleSize + thread_y * RADIUS + thread_x * radiusDiv + i];
		

	//New method that removes the if statements completely
	//Not sure if it works yet
			avg_value = calculate_averageValue(avg_value,avg_valueOrtho);
			avg_valueOrtho = calculate_averageValueOrtho(avg_valueOrtho);

	
			//Storing the averages back into the shared memory array
			averages[thread_y * RADIUS + thread_x * radiusDiv + i] = avg_value;
			averages[RADIUS * angleSize + thread_y * RADIUS + thread_x * radiusDiv + i] = avg_valueOrtho;


			//printf("%d %d %f %f\n",thread_y,thread_x,avg_value,avg_valueOrtho);
		}
	
	//}
//Works till here!

		__syncthreads();

	}

	//--------------Transposing the matrix to get the smallest number from the columns-----------------------------------//
//*********************
//TRANSPOSE NOT NECESSARY IF USING ONE THREAD PER COLUMN
//*********************
/*



		int i,j,k;
		for(radIdx=0;radIdx<radiusDiv;radIdx++){
		    //for(int n = 0; n<N*M; n++) {
			//int i = n/N;
			//int j = n%N;
			//dst[n] = src[M*j + i];
		    //}
			i = (thread_y * RADIUS + thread_x * radiusDiv + radIdx)/angleSize;
			j = (thread_y * RADIUS + thread_x * radiusDiv + radIdx)%angleSize;
			k = RADIUS * j + i;
			//Transposing the matrix but fitting it into the same matrix; This is done to get min values from each RADIUS
			averages[RADIUS * angleSize + thread_y * RADIUS + thread_x * radiusDiv + radIdx] = averages[2 * RADIUS * angleSize + k];

		}
		__syncthreads();	

		//Now the second has the transposed last matrix such that all the values fit;
		//The first matrix has averaged Ortho values (avg_valueOrtho); 
		//The last matrix still has the averaged values (avg_values);
	
	}
*/
//------------------------------------------------------------------------------------------------------------//
	//Finding the minimum over the columns
	//if(threadIdx.x < (dividedRadius * angleSize))
	//{

			
	if(threadIdx.x < RADIUS)
	{

		avg_value = averages[threadIdx.x];

		avg_valueSum = angle[0];
		
		avg_valueOrtho = averages[RADIUS * angleSize + threadIdx.x];


		//In the non shared memory version this looks like:
		/*
		if(avg_value < variance[j]) {
			variance[j] = avg_value;
			orientation[j] = angle[i];
			ortho[j] = avg_valueOrtho;		
		}
		*/

		for(i=0;i<angleSize;i++)
		{			
			//printf("Radius Angle current_value conditionalValue %d %d %f %f\n",threadIdx.x,i,averages[i*RADIUS + threadIdx.x],condition_if_lessthan(averages[i*RADIUS+threadIdx.x],avg_value,averages[i*RADIUS + threadIdx.x],avg_value));				

			//Same condition for each case : if(avg_value < variance[j]);  Therefore the floorf, or the equation part does not change just the variables do
			avg_valueSum = condition_if_lessthan(averages[i*RADIUS+threadIdx.x],avg_value,angle[i],avg_valueSum);
			avg_valueOrtho = condition_if_lessthan(averages[i*RADIUS+threadIdx.x],avg_value,averages[RADIUS * angleSize + i * RADIUS + threadIdx.x],avg_valueOrtho);
			avg_value = condition_if_lessthan(averages[i*RADIUS+threadIdx.x],avg_value,averages[i*RADIUS + threadIdx.x],avg_value);


//avg_value same for this and the non-shared memory code 
//Needed to move the avg_value calculation below others in the code above
//printf("%d %d %f %f %f\n",threadIdx.x,i,avg_value,avg_valueSum,avg_valueOrtho);	
//**Works till here
			

			/*
			avg_value = floorf(averages[i*RADIUS+threadIdx.x]/avg_value) * avg_value + (1.0 -  floorf(averages[i*RADIUS+threadIdx.x]/avg_value) ) * averages[i*RADIUS + threadIdx.x];
			avg_valueSum = floorf(averages[i*RADIUS+threadIdx.x]/avg_value) * avg_valueSum +  (1.0 -  floorf(averages[i*RADIUS+threadIdx.x]/avg_value) ) * angle[i];
			avg_valueOrtho = floorf(averages[i*RADIUS +threadIdx.x]/avg_value) * avg_valueOrtho + (1.0 -  floorf(averages[i*RADIUS+threadIdx.x]/avg_value) ) * averages[RADIUS * angleSize + i * RADIUS + threadIdx.x];
			*/

				
		}
//Incorrect order of division; avg_value/avg_valueOrtho was put in instead of acg_valueOrtho/avg_value
		printf("%d %d %d %d %f %f %f\n",threadIdx.x,i,dataIdx_x,dataIdx_y,avg_value,avg_valueSum,avg_valueOrtho);	
		anisotropy[dataIdx_y * XSIZE * RADIUS + dataIdx_x * RADIUS + threadIdx.x] = avg_valueOrtho/avg_value; 
		azimuth[dataIdx_y * XSIZE * RADIUS + dataIdx_x * RADIUS + threadIdx.x] = avg_valueSum * 180/3.14159; 
		
		__syncthreads();

/*
		anisotropy[dataIdx_y * XSIZE * RADIUS + dataIdx_x * RADIUS + threadIdx.x] = avg_value/avg_valueOrtho; 
		azimuth[dataIdx_y * XSIZE * RADIUS + dataIdx_x * RADIUS + threadIdx.x] = avg_valueSum * 180/3.14159; 
*/

	//	printf("dataIdx_x dataIdx_y Radius Ani Azi %d %d %d %f %f\n",dataIdx_x,dataIdx_y,threadIdx.x,avg_value/avg_valueOrtho,avg_valueSum * 180/3.14159);
		
	}
	
/*

		if(threadIdx.x < RADIUS)
		{
			//Minimum value stored in avg_value to save on registers (local vars)
			avg_value = averages[threadIdx.x];
			//Orientation stored in sum_value to save on registers (local vars)
			sum_value = angle[0];
			//Ortho value stored in avg_valueOrtho to save on registers (local vars)
			avg_valueOrtho = averages[RADIUS * angleSize + threadIdx.x];


			for(i=1;i<angleSize;i++){

				if(averages[i * RADIUS + threadIdx.x] < avg_value){
					avg_value = averages[i*RADIUS + threadIdx.x];
					sum_value = angle[i];
					avg_valueOrtho = averages[RADIUS * angleSize + i * RADIUS + threadIdx.x];
				}
			}


		}

	//}

		__syncthreads();

	//if(threadIdx.x < (dividedRadius * angleSize))
	//{
		if(threadIdx.x < RADIUS)
		{
			anisotropy[dataIdx_y * XSIZE * RADIUS + dataIdx_x * RADIUS + threadIdx.x] = avg_value/avg_valueOrtho; 
			azimuth[dataIdx_y * XSIZE * RADIUS + dataIdx_x * RADIUS + threadIdx.x] = sum_value * 180/3.14159; 
		
		}
		
	}//End of the long if(threadIdx.x <(dividedRadius * angleSize) condition
*/
	//------------------------------------------------------------------------------------------------------------//

	return;		

 
}

//--------------------------------------END OF KERNEL-----------------------------------------------------------//



//--------------------------------------Handle Error()-----------------------------------------------------------//

static void HandleError( cudaError_t err,const char *file, int line) 
{
	if (err != cudaSuccess) 
	{
	        fprintf( stderr,"%s in %s at line %d\n", cudaGetErrorString( err ),file, line );
		exit(err);
    	}
}


//--------------------------------------Get_GPU_devices()-----------------------------------------------------------//

int Get_GPU_devices()
{
	cudaDeviceProp prop;
	int whichDevice,DeviceCount;
	HANDLE_ERROR(cudaGetDevice(&whichDevice));
	HANDLE_ERROR(cudaGetDeviceProperties(&prop,whichDevice));

	if(!prop.deviceOverlap){
		printf("Device does not handle overlaps so streams are not possible\n");
	return 0;
	}

	DeviceCount = 0;
	
	HANDLE_ERROR(cudaGetDeviceCount(&DeviceCount));
	if(DeviceCount > 0){ 
		printf("%d Devices Found\n",DeviceCount);
	}else{
		printf("No devices found or error in reading the number of devices\n");
		return 0;
	}
	
	for(int i = 0;i<DeviceCount;i++){
		cudaDeviceProp properties;
		HANDLE_ERROR(cudaGetDeviceProperties(&properties,i));
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", properties.name);
		printf("  Device Global Memory size: %zd MB \n",properties.totalGlobalMem/1000000);
		printf("\n");

	}

	return DeviceCount;
}



//#################################################################################################################################//
//#################################################################################################################################//
//#################################################################################################################################//

//--------------------------------------Main()-----------------------------------------------------------//
int main(int argc,char* argv[])
//int main()
{

	
	char FileName[20];
	char delimiterStr[10];
	char delimiter;
	int RADIUS;
	int WINDOW_SIZE;	
	//Size of angle array
	int ANGLESIZE = 36;	
	
	if(argc != 9){
		printf("\tArguments needed = 9; Provided = %d\n",argc);
                printf("Usage: ./Executable -i InputDataFileName -d Delimiter -r Radius -w WindowSize\n");
		printf("Exiting program\n");
		return 0;
	}


	int option;

	while ((option = getopt(argc, argv,"i:d:r:w:")) != -1) {
		switch (option) {
                        case 'i' : strcpy(FileName,optarg);
                            break;
                        case 'd' : strcpy(delimiterStr,optarg);
                            break;
                        case 'r' : RADIUS  = atoi(optarg);
                            break;
                        case 'w' : WINDOW_SIZE = atoi(optarg);
                            break;
                        default: printf("Usage: Executable -i InputDataFileName -d Delimiter -r Radius -w WindowSize\n");
                            exit(EXIT_FAILURE);
                }
         }


	//In the future use optarg
	if(strcmp(delimiterStr,"space")==0){
		delimiter = ' ';
	}
	else if(strcmp(delimiterStr,"Space")==0){
		delimiter = ' ';
	}
	else if(strcmp(delimiterStr,"tab")==0){
		delimiter = '\t';
	}
	else if(strcmp(delimiterStr,"Tab")==0){
		delimiter = '\t';
	}
	else{
		delimiter = delimiterStr[0];
	}
	
	printf("Delimiter: %c\n",delimiter);
	//return 0;

	//#undef RADIUS
	//#define RADIUS atoi(argv[3])

	//RADIUS = tmp;
	printf("Radius is %d\n",RADIUS);
	printf("AngleSize is %d\n",ANGLESIZE);
	printf("Input file name is: %s\n",FileName);


	if(RADIUS > 100){
		printf("Radius value cannot exceed 100\nExiting\n");
		return (-1);
	}
//-------------------------------------------------------------------------------------//	
	//File declarations and opening them
	FILE *datTxt1,*datTxt;
	FILE *outputAnisotropy00,*outputAnisotropy09,*outputAnisotropy24,*outputAnisotropy49,*outputAnisotropy99;
	FILE *outputAzimuth00,*outputAzimuth09,*outputAzimuth24,*outputAzimuth49,*outputAzimuth99; 
	
	FILE *outputAnisotropy04,*outputAzimuth04;
	FILE * inpCheck;


	datTxt1 = fopen(FileName,"r");	
	inpCheck = fopen("inpCheck.txt","w");

	if(inpCheck == NULL) {
		perror("Cannot open inpcheck.txt file");
		return (-1);
	}
		
	if(datTxt1 == NULL) {
		printf("Cannot open file: %s  \nCheck if file exists.\n",argv[1]);
		exit(1);
	}

//-------------------------------------------------------------------------------------//
//				Setting Up Output Filenames
//-------------------------------------------------------------------------------------//

	char AniFirst[80],AniFive[80],AniTen[80],AniTwentyFive[80],AniFifty[80],AniLast[80];
	char AziFirst[80],AziFive[80],AziTen[80],AziTwentyFive[80],AziFifty[80],AziLast[80];
	
	strcpy(AniFirst,"Out_Ani_First_");
	strcpy(AniFive,"Out_Ani_Five_");
	strcpy(AniTen,"Out_Ani_Ten_");
	strcpy(AniTwentyFive,"Out_Ani_TwentyFive_");
	strcpy(AniFifty,"Out_Ani_Fifty_");
	strcpy(AniLast,"Out_Ani_Last_");

	strcat(AniFirst,FileName);
	strcat(AniFive,FileName);
	strcat(AniTen,FileName);
	strcat(AniTwentyFive,FileName);
	strcat(AniFifty,FileName);
	strcat(AniLast,FileName);

	strcpy(AziFirst,"Out_Azi_First_");
	strcpy(AziFive,"Out_Azi_Five_");
	strcpy(AziTen,"Out_Azi_Ten_");
	strcpy(AziTwentyFive,"Out_Azi_TwentyFive_");
	strcpy(AziFifty,"Out_Azi_Fifty_");
	strcpy(AziLast,"Out_Azi_Last_");

	strcat(AziFirst,FileName);
	strcat(AziFive,FileName);
	strcat(AziTen,FileName);
	strcat(AziTwentyFive,FileName);
	strcat(AziFifty,FileName);
	strcat(AziLast,FileName);

	printf("Ani First is %s\n",AniFirst);
//-------------------------------------------------------------------------------------//


	outputAnisotropy00 = fopen(AniFirst,"a");
	outputAnisotropy04 = fopen(AniFive,"a");
	outputAnisotropy09 = fopen(AniTen,"a");
	outputAnisotropy24 = fopen(AniTwentyFive,"a");
	outputAnisotropy49 = fopen(AniFifty,"a");
	outputAnisotropy99 = fopen(AniLast,"a");
	if((outputAnisotropy00 == NULL)||(outputAnisotropy09 == NULL)||(outputAnisotropy49 == NULL)||(outputAnisotropy99 == NULL)) {
		perror("Cannot open Anisotropy file");
		return (-1);
	}

	outputAzimuth00 = fopen(AziFirst,"a");
	outputAzimuth04 = fopen(AziFive,"a");
	outputAzimuth09 = fopen(AziTen,"a");
	outputAzimuth24 = fopen(AziTwentyFive,"a");
	outputAzimuth49 = fopen(AziFifty,"a");
	outputAzimuth99 = fopen(AziLast,"a");

	if((outputAzimuth00 == NULL)||(outputAzimuth09 == NULL)||(outputAzimuth49 == NULL)||(outputAzimuth99 == NULL)) {
		perror("Cannot open Azimuth file");
		return (-1);
	}

//-----------Getting total rows and columns in the data file---------------------------------------------------------------------------------------------------//

	long int XSIZE,YSIZE;
	XSIZE = 0;
	YSIZE = 0;
	long int i,j;

	//Counting number of columns(x)
	char* max_line;
	max_line = (char*)malloc(MAX_XSIZE_POSSIBLE);
	memset(max_line,'\0',sizeof(max_line));

	fgets(max_line,MAX_XSIZE_POSSIBLE,datTxt1)!=NULL; 
	while(*max_line !='\0'){
		if(*max_line == delimiter){
			XSIZE++;
		}
		max_line++;
	}
	
	XSIZE+=1;
	
	//Counting number of rows(y)
	do{
		i = fgetc(datTxt1);
		if(i == '\n') YSIZE++;
	}while(i != EOF);
	YSIZE+=1;
	
	printf("(XSIZE,YSIZE)::(%ld,%ld)\n",XSIZE,YSIZE);

	datTxt = fopen(FileName,"r");
	if(datTxt == NULL) {
		printf("Cannot open file: %s\nCheck if file exists\n",argv[1]);
		exit(1);
	}
//-----------------------Checking if the data size fits the memory of the GPU----------------------------------------------------------------------------------------//

	printf("(XSIZE,YSIZE):(%ld,%ld)\n",XSIZE,YSIZE);
	//printf("Maximum size possible = %f\nTotal size of current data(XSIZE * YSIZE) = %zd\n",MAX_XSIZE_POSSIBLE,XSIZE * YSIZE);
	//(MAX_XSIZE_POSSIBLE - XSIZE*YSIZE >0)? printf("There is enough memory for the computation\n"):printf("There is not enough memory and may result in incorrect results\n");

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------//


	float* data;

	data = (float*)malloc(YSIZE * XSIZE * sizeof(float));

	//XSIZE ints in a row which are max of 5 digits
	//with a space in the front and the back and space
	//between each number 
	char *startPtr,*endPtr;
	char line[XSIZE * 10 +2+(XSIZE-1)];
	memset(line, '\0', sizeof(line));
	float Value;
	i = 0;
	j = 0;
	//Assuming each number in the data set has a max of 7 characters
	char tempVal[5];
	memset(tempVal,'\0',sizeof(tempVal));

	printf("Reading the data file.\n");
	while(fgets(line,XSIZE *10 + 2 + (XSIZE-1),datTxt)!=NULL) {	
		//printf("Working2\n");
		startPtr = line;	
		for(i=0;i<XSIZE;i++) {
			Value = 0;
			memset(tempVal,'\0',sizeof(tempVal));		
			if(i != (XSIZE - 1)) {	
				endPtr = strchr(startPtr,delimiter);
				strncpy(tempVal,startPtr,endPtr-startPtr); 
				Value = atof(tempVal);
				*(data + j * XSIZE + i) = Value;
				fprintf(inpCheck,"%f ",Value);
				//printf("(j,i)::(%d,%d)\n",j,i);
				//printf("Column %d\n",i);

				endPtr = endPtr + 1;
				startPtr = endPtr;
			}	
			else if(i == (XSIZE - 1)){
				strcpy(tempVal,startPtr);
				Value = atof(tempVal);
				*(data + j * XSIZE + i) = Value;
				fprintf(inpCheck,"%f\n",Value);
					
			//	printf("(j,i)::(%d,%d)\n",j,i);
		
			}
		}
		
		j++;
	}
	printf("Closing the inputdata text files. \n");
	fclose(datTxt);
	fclose(datTxt1);
	fclose(inpCheck);

	printf("Done data[%zd][%zd] = %f\n",j-1,i-1,*(data + (j-1) * XSIZE + (i-1)));	
	printf("Working File IO\n");


	HANDLE_ERROR(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 10000000));
//-----------Getting the number of devices and their sizes------------------------------------------------//
        const int DeviceCount = Get_GPU_devices();


//------------------Initializing the structures that will hold GPU data-----------------------------------//
	GPU_struct GPU_values[DeviceCount];
	//HANDLE_ERROR(cudaMallocHost((void**)&GPU_values,DeviceCount*sizeof GPU_values));
//--------------------------------------------------------------------------------------------------------//

	//XSIZE  = number of total columns
	//YSIZE = number of total rows

	//Variable that holds number of rows for each GPU 
	int tmpSize = 0;
	//offset holds either 2*RADIUS or RADIUS depending on the part of data
	int offset = 0;
	//sum of the total positions of the rows for each GPU in each iteration
	int pos = 0;
	//Actual position inside the data matrix
	size_t data_position = 0;

	tmpSize = YSIZE/DeviceCount;
	printf("Each GPU gets %d rows\n",tmpSize);

	//Iterating through the available devices upto the second last
	for(i = 0;i<DeviceCount - 1;i++){

		printf("\n######################## Device %d #############################\n",i);

		
		GPU_values[i].NumCols = XSIZE;	

		//The last device is taken care after getting the remaining rows
		if(i == 0){
			GPU_values[i].NumRows = tmpSize + RADIUS;
			GPU_values[i].size = (GPU_values[i].NumRows + RADIUS ) * XSIZE;
			printf("Number of rows are: %ld\n",GPU_values[i].NumRows);
		//Sections in between
			printf("i is: %ld\n",i);
		}else{
			GPU_values[i].NumRows = tmpSize + 2 * RADIUS;
			GPU_values[i].size = (GPU_values[i].NumRows + 2*RADIUS) * XSIZE;
			printf("Number of rows are: %ld\n",GPU_values[i].size/XSIZE);
			//offset = RADIUS * -1;
		}
		printf("Size is: (GPU_values[%zd].NumRows + RADIUS) * XSIZE *sizeof(int) = (%ld + %d )*%ld *%ld =  %ld\n",i,GPU_values[i].NumRows,RADIUS,XSIZE,sizeof(float),GPU_values[i].size*sizeof(float));	
	}

	//---------------------Allocating number of rows to the last device--------------------------------//
	printf("\n########################Device %d ############################\n",DeviceCount -1);

	//Store the remaining rows in the last GPU
	GPU_values[DeviceCount - 1].NumRows = YSIZE - (tmpSize * (DeviceCount - 1)) + RADIUS;
	GPU_values[i].NumCols = XSIZE;	
	GPU_values[DeviceCount - 1].size = (GPU_values[DeviceCount - 1].NumRows + RADIUS) * XSIZE;

//--------------------If only a single GPU was found--------------------------------------------//
	if(DeviceCount == 1){
		GPU_values[0].NumRows = YSIZE;
		GPU_values[0].NumCols = XSIZE;
		GPU_values[0].size = YSIZE * XSIZE;

		printf("NumRows: %d, NumCols: %d\n",GPU_values[0].NumRows,GPU_values[0].NumCols);
	}

	//----------------------------------------------------------------------------------------------//
	printf("Number of rows are: %ld\n",GPU_values[DeviceCount - 1].NumRows);
	printf("i is: %ld\n",DeviceCount - 1);

	//----------------------------------------------------------------------------------------------//


	int numSegments = 1;
	
	for(i = 0;i<DeviceCount;i++){

		if((i==0) || (i==DeviceCount-1)){
			numSegments = 1;
		}else{
			numSegments = 2;
		}


		printf("\n########################Device %d #############################\n",i);
		printf("Radius is %d\n",RADIUS);
		//-----------------Matrix Allocations----------------------------//
		HANDLE_ERROR(cudaSetDevice(i));
		HANDLE_ERROR(cudaStreamCreate(&GPU_values[i].stream));
		HANDLE_ERROR(cudaDeviceSetLimit(cudaLimitMallocHeapSize, (size_t)(GPU_values[i].size *sizeof(int) + ANGLESIZE * sizeof(float) + 8*GPU_values[i].size * RADIUS/RADSTEP * sizeof(float))));

		HANDLE_ERROR(cudaMalloc((void**)&GPU_values[i].d_data,GPU_values[i].size * sizeof(int)));	
		HANDLE_ERROR(cudaMalloc((void**)&GPU_values[i].d_angle,ANGLESIZE * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&GPU_values[i].d_anisotropy,GPU_values[i].size * RADIUS/RADSTEP * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&GPU_values[i].d_azimuth,GPU_values[i].size * RADIUS/RADSTEP * sizeof(float)));

		HANDLE_ERROR(cudaMallocHost((void**)&GPU_values[i].h_data,GPU_values[i].size * sizeof(int)));
		HANDLE_ERROR(cudaMallocHost((void**)&GPU_values[i].h_angle,ANGLESIZE * sizeof(float)));
		HANDLE_ERROR(cudaMallocHost((void**)&GPU_values[i].h_anisotropy,GPU_values[i].size * RADIUS/RADSTEP * sizeof(float)));
		HANDLE_ERROR(cudaMallocHost((void**)&GPU_values[i].h_azimuth,GPU_values[i].size * RADIUS/RADSTEP * sizeof(float)));

		//---------------Initialization of data arrays for each GPU---------// 
		//Populating the angle array
		for(j=0;j<ANGLESIZE;j++) {
			GPU_values[i].h_angle[j] = j * 5 * 3.14159/180;	
		}

		data_position = (pos + offset ) * XSIZE;

		printf("pos = %d,data_position after sub Index = %zd\n",pos,data_position);

		//Initializing the data arrays in each of the gpu with portions of the main data
		for(j=0;j<GPU_values[i].size;j++){
			GPU_values[i].h_data[j] = *(data + data_position+j);
		}
		
		printf("Data array assigned \n");
		offset = RADIUS * -1;
		pos+=GPU_values[i].NumRows-numSegments*RADIUS;	
	}

	//Shared memory only holds the anisotropy data for each point in the (radius,anglesize) grid
	size_t SharedMemSize = 2 * RADIUS * ANGLESIZE * sizeof(float);
	int threadsPBlock_X;
	int threadsPBlock_Y; 
	//Radius is divided by radiusDiv to make sure number of threads per block is less than max (1024)
	int radiusDiv;
	int tempSize;

//***NOTE***
	//Lowest number of threads per block that can be if only changing the X axis is : (32,64,1)
	//And 32 * 64 = 2048 which is greater than 1024!!
	//Both the RADIUS and the ANGLESIZE has to change!
	//BUT, does the total number of threads have to be multiple of one warp size (32) or 
	//	threads in each dimension have to be a multiple of one warp size (32)



//	(RADIUS % 32) == 0?:threadsPBlock_X = RADIUS:threadsPBlock_X = 32 * (RADIUS/32 + 1);
//	(ANGLESIZE % 32) == 0?:threadsPBlock_Y = ANGLESIZE;threadsPBlock_Y = 32 * (ANGLESIZE/32 + 1);


	//Creating a one dimensional thread block	
	threadsPBlock_Y = 1;

	//Getting the maximum possible threads per block as it cannot exceed 1024
	if( RADIUS * ANGLESIZE > 1024){

		for(radiusDiv=5; radiusDiv<10; radiusDiv++){

			if(RADIUS % radiusDiv == 0) {

				tempSize = RADIUS/radiusDiv * ANGLESIZE;
				if((tempSize % 32) != 0){
					tempSize = 32 * (tempSize/32 + 1);
				}

				if(tempSize < 1024){
					//threadsPBlock_X = RADIUS/radiusDiv;
					threadsPBlock_X = tempSize;
					break;
				}			
			}	
		}
	}
	//If the total threads per block is less than 1024
	else{
		//Since the total threads is less than 1024 there is no need to divide the RADIUS		
		radiusDiv = 1;
		threadsPBlock_X = RADIUS * ANGLESIZE;

		if(threadsPBlock_X % 32 != 0){
			threadsPBlock_X  = 32 * (threadsPBlock_X/32 + 1);
		}				
	}


	printf("radiusDiv is: %d\n",radiusDiv);
	printf("Total Threads per block is: %d \n",threadsPBlock_X);
	printf("SharedMemSize: %ld, threadsPBlock_X: %d, threadsPBlock_Y: %d\n",SharedMemSize,threadsPBlock_X,threadsPBlock_Y);
	printf("GridX: %d, GridY: %d\n",GPU_values[0].NumCols - 2* RADIUS,GPU_values[0].NumRows - 2* RADIUS);

	
	for(i=0;i<DeviceCount;i++){
		printf("\n########################Device %d #############################\n",i);
		HANDLE_ERROR(cudaSetDevice(i));

		//-----------------Sending data to GPU----------------------//
		HANDLE_ERROR(cudaMemcpyAsync(GPU_values[i].d_data,GPU_values[i].h_data,GPU_values[i].size * sizeof(int),cudaMemcpyHostToDevice,GPU_values[i].stream));
		HANDLE_ERROR(cudaMemcpyAsync(GPU_values[i].d_angle,GPU_values[i].h_angle,ANGLESIZE * sizeof(float),cudaMemcpyHostToDevice,GPU_values[i].stream));

		//----------------Kernel Variables---------------------//
		if((i==0) || (i==DeviceCount-1)){
			numSegments = 1;
		}else{
			numSegments = 2;

		}
		
		//For 1 GPU
		dim3 gridSize(GPU_values[i].NumCols - 2* RADIUS,GPU_values[i].NumRows - 2*RADIUS,1);
		dim3 blockSize(threadsPBlock_X,threadsPBlock_Y,1);

		//printf("GridSize(X,Y) = (%ld,%ld)\n",(GPU_values[i].NumCols + THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,(GPU_values[i].NumRows));

		//----------------Launching the Kernel---------------------//
		printf("Kernel Launch # : %d\n",i);
		getMatrix<<<gridSize,blockSize,SharedMemSize,GPU_values[i].stream>>>(GPU_values[i].d_data,GPU_values[i].d_angle,GPU_values[i].d_anisotropy,GPU_values[i].d_azimuth,GPU_values[i].NumCols,GPU_values[i].NumRows,RADIUS,ANGLESIZE,radiusDiv);
		//HANDLE_ERROR(cudaPeekAtLastError());

		//HANDLE_ERROR(cudaDeviceSynchronize());
		//getLastCudaError("Kernel failed \n");

	}

	for(i=0;i<DeviceCount;i++){

		HANDLE_ERROR(cudaDeviceSynchronize());
		//---------------Getting data back------------------------//
		HANDLE_ERROR(cudaMemcpyAsync(GPU_values[i].h_anisotropy,GPU_values[i].d_anisotropy,GPU_values[i].size * RADIUS/RADSTEP * sizeof(float),cudaMemcpyDeviceToHost,GPU_values[i].stream));
		HANDLE_ERROR(cudaMemcpyAsync(GPU_values[i].h_azimuth,GPU_values[i].d_azimuth,GPU_values[i].size * RADIUS/RADSTEP * sizeof(float),cudaMemcpyDeviceToHost,GPU_values[i].stream));

		printf("Device # %ld\n",i);
	}

	printf("\n");
	int z;
	//To offset the data by radius so that the read starts in the correct segment
	int offsetRadius = 0;

	for(z = 0;z<DeviceCount;z++){

		printf("\n###############################################################\n",z);
		printf("\n########################Device %d #############################\n",z);
		HANDLE_ERROR(cudaSetDevice(z));
		HANDLE_ERROR(cudaStreamSynchronize(GPU_values[z].stream));

		printf("Rows: %ld,Cols: %ld\n",GPU_values[z].NumRows,GPU_values[z].NumCols);
		printf("Radius is: %d\n",RADIUS);

		if((z==0) || z==(DeviceCount-1)){
			offsetRadius = 0;
		}else{
			offsetRadius = 1;
		}

		for(j=0;j<GPU_values[z].NumRows ;j++) {

			for(i=0;i<GPU_values[z].NumCols ;i++) {

				if((j>(GPU_values[z].NumRows - RADIUS - 1))||(j<(RADIUS))) continue;
				if((i>(GPU_values[z].NumCols - RADIUS - 1))||(i<(RADIUS))) continue;

				//printf("Col:%ld,Row: %ld\n",i,j);
				//If last element in the row
				if (i == (GPU_values[z].NumCols  - RADIUS - 1)) {
					fprintf(outputAnisotropy00,"%f",GPU_values[z].h_anisotropy[j * GPU_values[z].NumCols  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + 0]);
					fprintf(outputAzimuth00,"%f",GPU_values[z].h_azimuth[j * GPU_values[z].NumCols * RADIUS/RADSTEP + i * RADIUS/RADSTEP + 0]);
					fprintf(outputAnisotropy00,"\n");
					fprintf(outputAzimuth00,"\n");

					fprintf(outputAnisotropy04,"%f",GPU_values[z].h_anisotropy[j * GPU_values[z].NumCols  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + 5]);
					fprintf(outputAzimuth04,"%f",GPU_values[z].h_azimuth[j * GPU_values[z].NumCols * RADIUS/RADSTEP + i * RADIUS/RADSTEP + 5]);
					fprintf(outputAnisotropy04,"\n");
					fprintf(outputAzimuth04,"\n");

					fprintf(outputAnisotropy09,"%f",GPU_values[z].h_anisotropy[j * GPU_values[z].NumCols  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + 10]);
					fprintf(outputAzimuth09,"%f",GPU_values[z].h_azimuth[j * GPU_values[z].NumCols * RADIUS/RADSTEP + i * RADIUS/RADSTEP + 10]);
					fprintf(outputAnisotropy09,"\n");
					fprintf(outputAzimuth09,"\n");

					fprintf(outputAnisotropy24,"%f",GPU_values[z].h_anisotropy[j * GPU_values[z].NumCols  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + RADIUS/4 - 1]);
					fprintf(outputAzimuth24,"%f",GPU_values[z].h_azimuth[j * GPU_values[z].NumCols * RADIUS/RADSTEP + i * RADIUS/RADSTEP + RADIUS/4 - 1]);
					fprintf(outputAnisotropy24,"\n");
					fprintf(outputAzimuth24,"\n");

					fprintf(outputAnisotropy49,"%f",GPU_values[z].h_anisotropy[j * GPU_values[z].NumCols  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + RADIUS/2 - 1]);
					fprintf(outputAzimuth49,"%f",GPU_values[z].h_azimuth[j * GPU_values[z].NumCols * RADIUS/RADSTEP + i * RADIUS/RADSTEP + RADIUS/2 - 1]);
					fprintf(outputAnisotropy49,"\n");
					fprintf(outputAzimuth49,"\n");

					fprintf(outputAnisotropy99,"%f",GPU_values[z].h_anisotropy[j * GPU_values[z].NumCols  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + RADIUS - 1]);
					fprintf(outputAzimuth99,"%f",GPU_values[z].h_azimuth[j * GPU_values[z].NumCols * RADIUS/RADSTEP + i * RADIUS/RADSTEP + RADIUS - 1]);
					fprintf(outputAnisotropy99,"\n");
					fprintf(outputAzimuth99,"\n");

					
				}
				else {
					fprintf(outputAnisotropy00,"%f",GPU_values[z].h_anisotropy[j * GPU_values[z].NumCols  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + 0]);
					fprintf(outputAzimuth00,"%f",GPU_values[z].h_azimuth[j * GPU_values[z].NumCols  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + 0]);
					fprintf(outputAnisotropy00,"\t");
					fprintf(outputAzimuth00,"\t");


					fprintf(outputAnisotropy04,"%f",GPU_values[z].h_anisotropy[j * GPU_values[z].NumCols  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + 5]);
					fprintf(outputAzimuth04,"%f",GPU_values[z].h_azimuth[j * GPU_values[z].NumCols * RADIUS/RADSTEP + i * RADIUS/RADSTEP + 5]);
					fprintf(outputAnisotropy04,"\t");
					fprintf(outputAzimuth04,"\t");

					fprintf(outputAnisotropy09,"%f",GPU_values[z].h_anisotropy[j * GPU_values[z].NumCols  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + 10]);
					fprintf(outputAzimuth09,"%f",GPU_values[z].h_azimuth[j * GPU_values[z].NumCols * RADIUS/RADSTEP + i * RADIUS/RADSTEP + 10]);
					fprintf(outputAnisotropy09,"\t");
					fprintf(outputAzimuth09,"\t");
	

					fprintf(outputAnisotropy24,"%f",GPU_values[z].h_anisotropy[j * GPU_values[z].NumCols  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + RADIUS/4 - 1]);
					fprintf(outputAzimuth24,"%f",GPU_values[z].h_azimuth[j * GPU_values[z].NumCols * RADIUS/RADSTEP + i * RADIUS/RADSTEP + RADIUS/4 - 1]);
					fprintf(outputAnisotropy24,"\t");
					fprintf(outputAzimuth24,"\t");

					fprintf(outputAnisotropy49,"%f",GPU_values[z].h_anisotropy[j * GPU_values[z].NumCols  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + RADIUS/2 - 1]);
					fprintf(outputAzimuth49,"%f",GPU_values[z].h_azimuth[j * GPU_values[z].NumCols * RADIUS/RADSTEP + i * RADIUS/RADSTEP + RADIUS/2 - 1]);
					fprintf(outputAnisotropy49,"\t");
					fprintf(outputAzimuth49,"\t");

					fprintf(outputAnisotropy99,"%f",GPU_values[z].h_anisotropy[j * GPU_values[z].NumCols  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + RADIUS - 1]);
					fprintf(outputAzimuth99,"%f",GPU_values[z].h_azimuth[j * GPU_values[z].NumCols * RADIUS/RADSTEP + i * RADIUS/RADSTEP + RADIUS - 1]);
					fprintf(outputAnisotropy99,"\t");
					fprintf(outputAzimuth99,"\t");
	
				}
					
			}
		}
	}	


	for(i=0;i<DeviceCount;i++){
		//------------------------------------Freeing data---------------------------------------//
		HANDLE_ERROR(cudaSetDevice(i));	
		HANDLE_ERROR(cudaStreamSynchronize(GPU_values[i].stream));
		HANDLE_ERROR(cudaDeviceSynchronize());

		HANDLE_ERROR(cudaFree(GPU_values[i].d_anisotropy));
		HANDLE_ERROR(cudaFree(GPU_values[i].d_azimuth));
		HANDLE_ERROR(cudaFree(GPU_values[i].d_data));
		HANDLE_ERROR(cudaFree(GPU_values[i].d_angle));

		HANDLE_ERROR(cudaFreeHost(GPU_values[i].h_data));
		HANDLE_ERROR(cudaFreeHost(GPU_values[i].h_angle));
		HANDLE_ERROR(cudaFreeHost(GPU_values[i].h_anisotropy));
		HANDLE_ERROR(cudaFreeHost(GPU_values[i].h_azimuth));

		HANDLE_ERROR(cudaStreamDestroy(GPU_values[i].stream));

		cudaDeviceReset();
			
	}

	/*for(i=0;i<DeviceCount;i++){
		HANDLE_ERROR(cudaSetDevice(i));
		HANDLE_ERROR(cudaFreeHost(GPU_values[i].h_data));
		cudaDeviceReset();
	}*/

	
	
	free(data);

	fclose(outputAnisotropy00);
	fclose(outputAnisotropy04);
	fclose(outputAnisotropy09);
	fclose(outputAnisotropy24);
	fclose(outputAnisotropy49);
	fclose(outputAnisotropy99);

	fclose(outputAzimuth00);
	fclose(outputAzimuth04);
	fclose(outputAzimuth09);
	fclose(outputAzimuth24);
	fclose(outputAzimuth49);
	fclose(outputAzimuth99);

	return 0;
}
		
		
