




//Input file: space delimited

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include "topographic_anisotropy_largerGrid.h"

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

//Size of the GPU memory
#define GPU_MEMSIZE_GB		2

//For case in which XSIZE = 1201 and YSIZE = 801
#define GLOBAL_MEM_USE_MB	773
#define MEM_USE_PER_THREAD_B	1280

//MAX_XSIZE_POSSIBLE is the maximum size of x or max number of columns if there is only one row
#define MAX_XSIZE_POSSIBLE	floor(((GPU_MEMSIZE_GB * 1000 - GLOBAL_MEM_USE_MB)*1000000)/MEM_USE_PER_THREAD_B) 


//#define XSIZE 		1201
//#define YSIZE			801


//Always have even number of radius;and divisible by 10
#define RADIUS			100
#define	RADSTEP			1
#define ANGLESIZE		36	//Size of angle array	

#define PI 			3.14159


#define THREADS_PER_BLOCK	512

//#define FILENAME	"Annie_coastDEM.txt"
//---------------------------Function declarations--------------------------------------------------------------------------//

__global__ void getMatrix(int* data,float* angle,float* anisotropy,float* azimuth,int XSIZE,int YSIZE);
int Get_GPU_devices();
static void HandleError(cudaError_t err,const char *file, int line);
inline cudaError_t checkCuda(cudaError_t result);
//--------------------------------------------------------------------------------------------------------------------------//

__global__ void getMatrix(int* data,float* angle,float* anisotropy,float* azimuth,int XSIZE,int YSIZE)
{

//	Thread indices
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
//----------------------------------------------------------------------------------------------------------------------------//	
	if((y>(YSIZE - RADIUS - 1))||(y<(RADIUS))) return;
	else if((x>(XSIZE - RADIUS - 1))||(x<(RADIUS))) return;
	else
	{
		//printf("%d,%d\n",XSIZE,YSIZE);
		//Actual computation
		int xrad,yrad,xradOrtho1,yradOrtho1,xradOneEighty,yradOneEighty,valueOneEighty;
		int valueOrtho1,valueOrtho2,xradOrtho2,yradOrtho2,i,j;

	
		float variance[RADIUS];
		float orientation[RADIUS];
		float ortho[RADIUS];


		float value,sum_value,avg_value;
		float sum_valueOrtho,avg_valueOrtho;

	//	Initializing declared variables
		sum_value = 0;
		avg_value = 0;
		sum_valueOrtho = 0;
		avg_valueOrtho = 0;

	//	Iniitalize variance, ortho, and orientation arrays with max float value SGR changed i<100 to i<RADIUS
		for(i=0;i<RADIUS;i++){
			variance[i] = FLT_MAX;
			ortho[i] = FLT_MAX;
			orientation[i] = FLT_MAX;
		}
				
		//Flipped
		for(i=0;i<ANGLESIZE;i++) {
			//Initializing to 0 so that the sum is zero everytime it starts
			sum_value = 0;
			sum_valueOrtho = 0;

			for(j = 0;j<RADIUS;j+=RADSTEP) {
	
				//Computation for angle of interest
				xrad = (int)lrintf(cosf(angle[i]) * (j+1) + x);	
				yrad = (int)lrintf(sinf(angle[i]) * (j+1) + y);	

				value = data[y * XSIZE + x]  - data[yrad * XSIZE + xrad];
				value = value * value;
				
				//One eighty angle computation
				xradOneEighty = (int)lrintf(cosf(angle[i]+PI) * (j+1) + x);	
				yradOneEighty = (int)lrintf(sinf(angle[i]+PI) * (j+1) + y);	
				
				valueOneEighty = data[y * XSIZE + x] - data[yradOneEighty * XSIZE + xradOneEighty];
				valueOneEighty = valueOneEighty * valueOneEighty;

				sum_value = sum_value + value + valueOneEighty;
				avg_value = sum_value/(2*(j+1)); //the average variance from scale 1 to scale j

				//Computation for values on angle orthogonal to angle of interest
				xradOrtho1 = (int)lrintf(cosf(angle[i]+PI/2) * (j+1) + x);	
				yradOrtho1 = (int)lrintf(sinf(angle[i]+PI/2) * (j+1) + y);	
				
				valueOrtho1 = data[y * XSIZE + x]  - data[yradOrtho1 * XSIZE + xradOrtho1];
				valueOrtho1 = valueOrtho1 * valueOrtho1;

				//One eighty ortho angle computation
				xradOrtho2 = (int)lrintf(cosf(angle[i]+PI*3/2) * (j+1) + x);	
				yradOrtho2 = (int)lrintf(sinf(angle[i]+PI*3/2) * (j+1) + y);	

				valueOrtho2 = data[y * XSIZE + x]  - data[yradOrtho2 * XSIZE + xradOrtho2];
				valueOrtho2 = valueOrtho2 * valueOrtho2;

				sum_valueOrtho = sum_valueOrtho + valueOrtho1 + valueOrtho2;
				avg_valueOrtho = sum_valueOrtho/(2*j+1);

				//Fail safe to ensure there is no nan or inf when taking anisotropy ratio, later on.			
				if(avg_value == 0) {
					if((avg_valueOrtho < 1) && (avg_valueOrtho > 0)) {
						avg_value = avg_valueOrtho;
					}
					else{
						avg_value = 1;
					}
				}

				if(avg_valueOrtho == 0) {
					avg_valueOrtho = 1;
				}
				
				//Determine if the variance is minimum compared to  others at scale j, if so record it and its angle i. If not, pass it
				if(avg_value < variance[j]) {
					variance[j] = avg_value;
					orientation[j] = angle[i];
					ortho[j] = avg_valueOrtho;		
				}	
			}
		}

		for(j=0;j<RADIUS;j+=RADSTEP){	
			anisotropy[y * XSIZE  * RADIUS/RADSTEP + x * RADIUS/RADSTEP + j] = ortho[j]/variance[j];
			azimuth[y * XSIZE  * RADIUS/RADSTEP + x * RADIUS/RADSTEP + j] = orientation[j] * 180/PI;
		}
	}
 
}

//--------------------------------------END OF KERNEL-----------------------------------------------------------//



//--------------------------------------Handle Error()-----------------------------------------------------------//

static void HandleError( cudaError_t err,const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
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


	if(argc == 1){
		printf("Not enough arguments\n");
		return 0;
	}
		


	#undef RADIUS
	#define RADIUS atoi(argv[2])
	//Setting the output buffer to 500MB
	size_t limit;
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 500 * 1024 * 1024);
	cudaDeviceGetLimit(&limit,cudaLimitPrintfFifoSize);

	//File declarations and opening them
	FILE *datTxt1,*datTxt;

	

	FILE * inpCheck;
	inpCheck = fopen("inpCheck.txt","w");
	if(inpCheck == NULL) {
		perror("Cannot open inpcheck.txt file");
		return (-1);
	}
	

	datTxt1 = fopen(argv[1],"r");	
	if(datTxt1 == NULL) {
		printf("Cannot open file: %s  \nCheck if file exists.\n",argv[1]);
		exit(1);
	}



//-----------Getting total rows and columns in the data file---------------------------------------------------------------------------------------------------//

	int XSIZE,YSIZE;
	XSIZE = 0;
	YSIZE = 0;
	long int i,j;

	//Counting number of columns(x)
	char* max_line;
	max_line = (char*)malloc(MAX_XSIZE_POSSIBLE);
	memset(max_line,'\0',sizeof(max_line));

	fgets(max_line,MAX_XSIZE_POSSIBLE,datTxt1)!=NULL; 
	while(*max_line)if(*max_line++ == ' ')++XSIZE;
	XSIZE+=1;
	
	//Counting number of rows(y)
	do{
		i = fgetc(datTxt1);
		if(i == '\n') YSIZE++;
	}while(i != EOF);
	YSIZE+=1;
	
	printf("(XSIZE,YSIZE)::(%d,%d)\n",XSIZE,YSIZE);

	datTxt = fopen(argv[1],"r");
	if(datTxt == NULL) {
		//printf("Cannot open file: %s\nCheck if file exists\n",argv[1]);
		exit(1);
	}
//-----------------------Checking if the data size fits the memory of the GPU----------------------------------------------------------------------------------------//

	printf("(XSIZE,YSIZE):(%d,%d)\n",XSIZE,YSIZE);
	//printf("Maximum size possible = %f\nTotal size of current data(XSIZE * YSIZE) = %zd\n",MAX_XSIZE_POSSIBLE,XSIZE * YSIZE);
	//(MAX_XSIZE_POSSIBLE - XSIZE*YSIZE >0)? printf("There is enough memory for the computation\n"):printf("There is not enough memory and may result in incorrect results\n");




//--------------------------------------------------------------------------------------------------------------------------------------------------------------------//


	int* data;

	data = (int*)malloc(YSIZE * XSIZE * sizeof(int));

	//XSIZE ints in a row which are max of 5 digits
	//with a space in the front and the back and space
	//between each number 
	char *startPtr,*endPtr;
	char line[XSIZE * 10 +2+(XSIZE-1)];
	memset(line, '\0', sizeof(line));
	int Value;
	i = 0;
	j = 0;
	//Assuming each number in the data set has a max of 7 characters
	char tempVal[5];
	memset(tempVal,'\0',sizeof(tempVal));

	printf("Working1\n");
	while(fgets(line,XSIZE *10 + 2 + (XSIZE-1),datTxt)!=NULL) {	
		//printf("Working2\n");
		startPtr = line;	
		for(i=0;i<XSIZE;i++) {
			Value = 0;
			memset(tempVal,'\0',sizeof(tempVal));		
			if(i != (XSIZE - 1)) {	
				endPtr = strchr(startPtr,' ');
				strncpy(tempVal,startPtr,endPtr-startPtr); 
				Value = atoi(tempVal);
				*(data + j * XSIZE + i) = Value;
				fprintf(inpCheck,"%d ",Value);
				//printf("(j,i)::(%d,%d)\n",j,i);

				endPtr = endPtr + 1;
				startPtr = endPtr;
			}	
			else if(i == (XSIZE - 1)){
				strcpy(tempVal,startPtr);
				Value = atoi(tempVal);
				*(data + j * XSIZE + i) = Value;
				fprintf(inpCheck,"%d\n",Value);
				//printf("(j,i)::(%d,%d)\n",j,i);
			}
		}
		
		j++;
	}

	fclose(datTxt);
	fclose(datTxt1);
	fclose(inpCheck);

	printf("Done data[%zd][%zd] = %d\n",j-1,i-1,*(data + 500 * XSIZE + 500));	
	printf("Working File IO\n");


//-----------Getting the number of devices and their sizes------------------------------------------------//
        int DeviceCount;
        DeviceCount = Get_GPU_devices();


//------------------Initializing the structures that will hold GPU data-----------------------------------//
	GPU_struct GPU_values[DeviceCount];
	//HANDLE_ERROR(cudaMallocHost((void**)&GPU_values,DeviceCount*sizeof GPU_values));
//--------------------------------------------------------------------------------------------------------//

	//XSIZE  = number of total columns
	//YSIZE = number of total rows

	//Variable that holds YSIZE initially. This changes as number of rows 
	//for each GPU is calculated
	int tmpSize = 0;
	//Variable needed to compute the total rows each GPU will have
	int count = 0;
	//offset holds either 2*RADIUS or RADIUS depending on the part of data
	int offset = 0;
	//sum of the total positions of the rows for each GPU in each iteration
	int pos = 0;
	//Actual position inside the data matrix
	size_t data_position = 0;

	tmpSize = YSIZE;
	count = DeviceCount;



	//Iterating through all the available devices
	for(i = 0;i<DeviceCount;i++){
		printf("\n########################Device %d #############################\n",i);

		//If the total rows are not exactly divisible by the number of GPUs; add 1
		if(tmpSize % count != 0){
			GPU_values[i].NumRows = (tmpSize/count) + 1;
			GPU_values[i].NumCols = XSIZE;
		}else{
			GPU_values[i].NumRows = tmpSize/count;
			GPU_values[i].NumCols = XSIZE;
		}	
		//Values change here as the num of rows for each gpu is 
		//calculated after each iteration
		tmpSize = tmpSize - GPU_values[i].NumRows;
		count--;
		printf("Row Value is: %d\n",GPU_values[i].NumRows);

		if((i == 0) ||(i == (DeviceCount -1))){
			GPU_values[i].size = (GPU_values[i].NumRows + RADIUS ) * XSIZE;	
			printf("i is: %d\n",i);
		//Sections in between
		}else{
			GPU_values[i].size = (GPU_values[i].NumRows + 2*RADIUS) * XSIZE;
			//offset = RADIUS * -1;
		}
		printf("Size is: GPU_values[%zd].NumRows + RADIUS = (%d + %d )*%d *%d =  %ld\n",i,GPU_values[i].NumRows,RADIUS,XSIZE,sizeof(float),GPU_values[i].size);	
	}





	for(i = 0;i<DeviceCount;i++){

		printf("\n########################Device %d #############################\n",i);

		//-----------------Matrix Allocations----------------------------//
		HANDLE_ERROR(cudaSetDevice(i));
		HANDLE_ERROR(cudaStreamCreate(&GPU_values[i].stream));

		HANDLE_ERROR(cudaMalloc((void**)&GPU_values[i].d_data,GPU_values[i].size *sizeof(int)));	
		HANDLE_ERROR(cudaMalloc((void**)&GPU_values[i].d_angle,ANGLESIZE * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&GPU_values[i].d_anisotropy,GPU_values[i].size * RADIUS/RADSTEP * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&GPU_values[i].d_azimuth,GPU_values[i].size * RADIUS/RADSTEP * sizeof(float)));

		HANDLE_ERROR(cudaMallocHost((void**)&GPU_values[i].h_data,GPU_values[i].size * sizeof(int)));
		HANDLE_ERROR(cudaMallocHost((void**)&GPU_values[i].h_angle,ANGLESIZE * sizeof(float)));
		HANDLE_ERROR(cudaMallocHost((void**)&GPU_values[i].h_anisotropy,GPU_values[i].size * RADIUS/RADSTEP * sizeof(float)));
		HANDLE_ERROR(cudaMallocHost((void**)&GPU_values[i].h_azimuth,GPU_values[i].size * RADIUS/RADSTEP * sizeof(float)));
		//---------------Initialization of data arrays for each GPU---------// 

		for(j=0;j<ANGLESIZE;j++) {
			GPU_values[i].h_angle[j] = j * 5 * PI/180;	
		}


		data_position = (pos + offset ) * XSIZE;

		printf("pos = %d,data_position after sub Index = %zd\n",pos,data_position);
		//Initializing the data arrays in each of the gpu with portions of the main data
		for(j=0;j<GPU_values[i].size;j++){
			GPU_values[i].h_data[j] = *(data + data_position+j);
			
			//if(j!=0 && j % 501 == 0) printf("\n");
			//printf("%d ",GPU_values[i].h_data[j]);
		}
		
		printf("Data array assigned \n");
		offset = RADIUS * -1;
		pos+=GPU_values[i].NumRows;	
	}


	for(i=0;i<DeviceCount;i++){
		
		HANDLE_ERROR(cudaSetDevice(i));

		//-----------------Sending data to GPU----------------------//
		HANDLE_ERROR(cudaMemcpyAsync(GPU_values[i].d_data,GPU_values[i].h_data,GPU_values[i].size * sizeof(int),cudaMemcpyHostToDevice,GPU_values[i].stream));
		HANDLE_ERROR(cudaMemcpyAsync(GPU_values[i].d_angle,GPU_values[i].h_angle,ANGLESIZE * sizeof(float),cudaMemcpyHostToDevice,GPU_values[i].stream));


		//----------------Kernel Variables---------------------//
		dim3 gridSize((GPU_values[i].NumCols + THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK ,(GPU_values[i].NumRows+RADIUS),1);
		dim3 blockSize(THREADS_PER_BLOCK,1,1);

		//----------------Launching the Kernel---------------------//
		getMatrix<<<gridSize,blockSize,0,GPU_values[i].stream>>>(GPU_values[i].d_data,GPU_values[i].d_angle,GPU_values[i].d_anisotropy,GPU_values[i].d_azimuth,GPU_values[i].NumCols,GPU_values[i].NumRows);

		//HANDLE_ERROR(cudaDeviceSynchronize());

		//---------------Getting data back------------------------//
		HANDLE_ERROR(cudaMemcpyAsync(GPU_values[i].h_anisotropy,GPU_values[i].d_anisotropy,GPU_values[i].size * RADIUS/RADSTEP * sizeof(float),cudaMemcpyDeviceToHost,GPU_values[i].stream));
		HANDLE_ERROR(cudaMemcpyAsync(GPU_values[i].h_azimuth,GPU_values[i].d_azimuth,GPU_values[i].size * RADIUS/RADSTEP * sizeof(float),cudaMemcpyDeviceToHost,GPU_values[i].stream));
	}



	for(i=0;i<DeviceCount;i++){
		//------------------------------------Freeing data---------------------------------------//
		HANDLE_ERROR(cudaSetDevice(i));	
		HANDLE_ERROR(cudaStreamSynchronize(GPU_values[i].stream));

		HANDLE_ERROR(cudaFree(GPU_values[i].d_anisotropy));
		HANDLE_ERROR(cudaFree(GPU_values[i].d_azimuth));
		HANDLE_ERROR(cudaFree(GPU_values[i].d_data));
		HANDLE_ERROR(cudaFree(GPU_values[i].d_angle));

		HANDLE_ERROR(cudaFreeHost(GPU_values[i].h_data));
		HANDLE_ERROR(cudaFreeHost(GPU_values[i].h_angle));
		HANDLE_ERROR(cudaFreeHost(GPU_values[i].h_anisotropy));
		HANDLE_ERROR(cudaFreeHost(GPU_values[i].h_azimuth));

		HANDLE_ERROR(cudaStreamDestroy(GPU_values[i].stream));

			
	}

	/*for(i=0;i<DeviceCount;i++){
		HANDLE_ERROR(cudaSetDevice(i));
		HANDLE_ERROR(cudaFreeHost(GPU_values[i].h_data));
		cudaDeviceReset();
	}*/

	
	
	free(data);
	
	return 0;
}
		
