



//Input file: space delimited

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <cuda.h>
#include <cuda_runtime.h>
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
//--------------------------------------------------------------------------------------------------------------------------//

//Current Usage:
//Global Memory: 773 MB


__global__ void getMatrix(int* data,float* angle,float* anisotropy,float* azimuth,int XSIZE,int YSIZE)
{
//	SGR I don't see where XSIZE or YSIZE are defined...
/*	
	//Actual computation
	int xrad,yrad,xradOrtho1,yradOrtho1,xradOneEighty,yradOneEighty,valueOneEighty;
	int valueOrtho1,valueOrtho2,xradOrtho2,yradOrtho2,i,j;

//	Hardwired to be at 100 Radius now. This needs to change!
	float variance[100];
	float orientation[100];
	float ortho[100];


	float value,sum_value,avg_value;
	float sum_valueOrtho,avg_valueOrtho;

//	Initializing declared variables
	sum_value = 0;
	avg_value = 0;
	sum_valueOrtho = 0;
	avg_valueOrtho = 0;
*/


	//for(int i = 0;i <YSIZE*XSIZE;i++){
	//	printf("%d\n",data[i]);
	//}

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
//--------------------------------------Main()-----------------------------------------------------------//
int main(int argc,char* argv[])
//int main()
{
	if(argc == 1){
		printf("Not enough arguments\n");
		return 0;
	}
		
//-----------Getting the number of devices and their sizes------------------------------------------------//
        int DeviceCount;
        DeviceCount = Get_GPU_devices();

	GPU_struct	GPU_values[DeviceCount];

	#undef RADIUS
	#define RADIUS atoi(argv[2])
	//Setting the output buffer to 500MB
	size_t limit;
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 500 * 1024 * 1024);
	cudaDeviceGetLimit(&limit,cudaLimitPrintfFifoSize);

	//File declarations and opening them
	FILE *datTxt1,*datTxt,*outputAnisotropy00,*outputAnisotropy09,*outputAnisotropy49,*outputAnisotropy99;
	FILE *outputAzimuth00,*outputAzimuth09,*outputAzimuth49,*outputAzimuth99; 
	

	FILE * inpCheck;
	inpCheck = fopen("inpCheck.txt","w");
	if(inpCheck == NULL) {
		perror("Cannot open inpcheck.txt file");
		return (-1);
	}
	

	datTxt1 = fopen(argv[1],"r");	
	//datTxt1 = fopen("Annie_coastDEM.txt","r");
	if(datTxt1 == NULL) {
		printf("Cannot open file: %s  \nCheck if file exists.\n",argv[1]);
		exit(1);
	}
	outputAnisotropy00 = fopen("outputDataAni_First.txt","w");
	outputAnisotropy09 = fopen("outputDataAni_Rad_div_10.txt","w");
	outputAnisotropy49 = fopen("outputDataAni_Rad_div_2.txt","w");
	outputAnisotropy99 = fopen("outputDataAni_Last.txt","w");
	if((outputAnisotropy00 == NULL)||(outputAnisotropy09 == NULL)||(outputAnisotropy49 == NULL)||(outputAnisotropy99 == NULL)) {
		perror("Cannot open Anisotropy file");
		return (-1);
	}

	outputAzimuth00 = fopen("outputDataAzi_First.txt","w");
	outputAzimuth09 = fopen("outputDataAzi_Rad_div_10.txt","w");
	outputAzimuth49 = fopen("outputDataAzi_Rad_div_2.txt","w");
	outputAzimuth99 = fopen("outputDataAzi_Last.txt","w");

	if((outputAzimuth00 == NULL)||(outputAzimuth09 == NULL)||(outputAzimuth49 == NULL)||(outputAzimuth99 == NULL)) {
		perror("Cannot open Azimuth file");
		return (-1);
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
	
	fclose(datTxt1);
	printf("(XSIZE,YSIZE)::(%d,%d)\n",XSIZE,YSIZE);

	datTxt = fopen(argv[1],"r");
//	datTxt = fopen("Annie_coastDEM.txt","r");
	if(datTxt == NULL) {
		//printf("Cannot open file: %s\nCheck if file exists\n",argv[1]);
		exit(1);
	}
//-----------------------Checking if the data size fits the memory of the GPU----------------------------------------------------------------------------------------//

	printf("(XSIZE,YSIZE):(%d,%d)\n",XSIZE,YSIZE);
	//printf("Maximum size possible = %f\nTotal size of current data(XSIZE * YSIZE) = %zd\n",MAX_XSIZE_POSSIBLE,XSIZE * YSIZE);
	//(MAX_XSIZE_POSSIBLE - XSIZE*YSIZE >0)? printf("There is enough memory for the computation\n"):printf("There is not enough memory and may result in incorrect results\n");




//--------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	int data[YSIZE * XSIZE];

//	int* data;

//	data = (int*)malloc(YSIZE * XSIZE * sizeof(int));

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
				data[j * XSIZE + i] = Value;
				fprintf(inpCheck,"%d ",Value);
				//printf("(j,i)::(%d,%d)\n",j,i);

				endPtr = endPtr + 1;
				startPtr = endPtr;
			}	
			else if(i == (XSIZE - 1)){
				strcpy(tempVal,startPtr);
				Value = atoi(tempVal);
				data[j * XSIZE + i] = Value;
				fprintf(inpCheck,"%d\n",Value);
				//printf("(j,i)::(%d,%d)\n",j,i);
			}
		}
		
		j++;
	}	
	printf("Done data[%zd][%zd] = %d\n",j-1,i-1,data[500 * XSIZE + 500]);	
	printf("Working File IO\n");

//----------Getting number of rows for each GPU-------------------------------------------------------------------------------------------------------------//
	//Variable that holds YSIZE initially. This changes as number of rows 
	//for each GPU is calculated
	int tmpSize = 0;
	//Variable needed to compute the total rows each GPU will have
	int count = 0;
	//size holds the total size of the data for each of the parts
	size_t size = 0;
	//size_orig holds the original size without the RADIUS added to it multiplied by RADIUS/RADSTEP
	size_t size_orig = 0;
	//offset holds either 2*RADIUS or RADIUS depending on the part of data
	int offset = 0;
	//sum of the total positions of the rows for each GPU in each iteration
	int pos = 0;
	//Actual position inside the data matrix
	size_t data_position = 0;
	//Store the total size of the angle array in bytes
	size_t angle_bytes = ANGLESIZE * sizeof(float);

	tmpSize = YSIZE;
	count = DeviceCount;

	printf("Tmpsize = %zd,Device Count = %d\n",tmpSize,DeviceCount);
	//Iterating through all the available devices
	for(i = 0;i<DeviceCount;i++){

		printf("\n########################Device %d #############################\n",i);

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

		size_orig = GPU_values[i].NumRows * XSIZE * RADIUS/RADSTEP *sizeof(float);

		if((i == 0) ||(i == (DeviceCount -1))){
			size = (GPU_values[i].NumRows + RADIUS ) * XSIZE * sizeof(float);	

			//Setting the offset into the data matrix
			if(i == 0){
				offset = 0;
			}else if(i == (DeviceCount - 1)){
				offset = RADIUS * -1;
			}
		//Sections in between
		}else{
			size = (GPU_values[i].NumRows + 2*RADIUS) * XSIZE * sizeof(float);
			offset = RADIUS * -1;
		}
		
		printf("Size is: GPU_values[%zd].NumRows + RADIUS = (%d + %d )*%d *%d =  %ld\n",i,GPU_values[i].NumRows,RADIUS,XSIZE,sizeof(float),size);

		//-----------------Matrix Allocations----------------------------//
		HANDLE_ERROR(cudaSetDevice(i));
		HANDLE_ERROR(cudaStreamCreate(&GPU_values[i].stream));

		HANDLE_ERROR(cudaMalloc((void**)&GPU_values[i].d_anisotropy,size_orig));
		HANDLE_ERROR(cudaMalloc((void**)&GPU_values[i].d_azimuth,size_orig));
		HANDLE_ERROR(cudaMalloc((void**)&GPU_values[i].d_data,size));	
		HANDLE_ERROR(cudaMalloc((void**)&GPU_values[i].d_angle,angle_bytes));
		printf("Cuda Malloc to GPU possible \n");


		HANDLE_ERROR(cudaMallocHost((void**)&GPU_values[i].h_anisotropy,size_orig));
		HANDLE_ERROR(cudaMallocHost((void**)&GPU_values[i].h_azimuth,size_orig));
		HANDLE_ERROR(cudaMallocHost((void**)&GPU_values[i].h_data,size));
		HANDLE_ERROR(cudaMallocHost((void**)&GPU_values[i].h_angle,angle_bytes));	

		printf("Cuda Malloc to CPU possible \n");

		
//---------------Initialization of data arrays for each GPU---------// 
		//Initially pos = 0 and offset = 0 so "if" statement is unnecessary

		for(j=0;j<ANGLESIZE;j++) {
			GPU_values[i].h_angle[j] = j * 5 * PI/180;		
		}


		size_t total_data;
		total_data = size/sizeof(float);

		printf("Total data size is %ld\n",total_data);

		printf("The size of the array is %d\n",sizeof(GPU_values[i].h_data));

		//if(i == 0){		
			data_position = (pos + offset) * XSIZE;
		//}else{
		//	data_position = (pos + offset + 1) * XSIZE;
		//}
		
		printf("pos = %d,data_position after sub Index = %zd\n",pos,data_position);
		//Initializing the data arrays in each of the gpu with portions of the main data
		for(j=0;j<(total_data);j++){
			GPU_values[i].h_data[j] = data[data_position+j];
			//if(i==1){
				//if(j!=0 && j % 501 == 0) printf("\n");
				//printf("%d ",GPU_values[i].h_data[j]);			
			//}
		}
		
		for(j=0;j<(total_data);j++){
				if((j!=0) && (j % 501 == 0)) printf("\n");
				printf("%d ",GPU_values[i].h_data[j]);			
			
		}
		printf("\n\n\n\n\n\n");
		



		printf("Data array assigned \n");
		offset = RADIUS * -1;
		//if(i == 0) {
		//	pos+=GPU_values[i].NumRows;
		//}else {
			pos+=GPU_values[i].NumRows;
		//}
	}
	printf("Working3\n");
	
//----------------Sending data to GPU;Launching Kernel and getting data back-----------------------//
	for(i=0;i<DeviceCount;i++){
		
		HANDLE_ERROR(cudaSetDevice(i));
		//HANDLE_ERROR(cudaStreamSynchronize(GPU_values[i].stream));

		if((i == 0) ||(i == (DeviceCount -1))){
			size = (GPU_values[i].NumRows + RADIUS) * XSIZE * sizeof(float);			
		//Sections in between
		}else{
			size = (GPU_values[i].NumRows + 2*RADIUS) * XSIZE * sizeof(float);
		}
//-----------------Sending data to GPU----------------------//
		HANDLE_ERROR(cudaMemcpyAsync(GPU_values[i].d_data,GPU_values[i].h_data,size,cudaMemcpyHostToDevice,GPU_values[i].stream));
		HANDLE_ERROR(cudaMemcpyAsync(GPU_values[i].d_angle,GPU_values[i].h_angle,angle_bytes,cudaMemcpyHostToDevice,GPU_values[i].stream));


//----------------Launching the Kernel---------------------//
		dim3 gridSize((XSIZE + THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK ,YSIZE,1);
		dim3 blockSize(THREADS_PER_BLOCK,1,1);

		getMatrix<<<gridSize,blockSize,0,GPU_values[i].stream>>>(GPU_values[i].d_data,GPU_values[i].d_angle,GPU_values[i].d_anisotropy,GPU_values[i].d_azimuth,XSIZE,YSIZE);
//---------------Getting data back------------------------//
		HANDLE_ERROR(cudaMemcpyAsync(GPU_values[i].h_anisotropy,GPU_values[i].d_anisotropy,size_orig,cudaMemcpyDeviceToHost,GPU_values[i].stream));
		HANDLE_ERROR(cudaMemcpyAsync(GPU_values[i].h_azimuth,GPU_values[i].d_azimuth,size_orig,cudaMemcpyDeviceToHost,GPU_values[i].stream));
	}
//------------------------------------Freeing data-----------------------------------------------------------------------//
	for(i=0;i<DeviceCount;i++){
		HANDLE_ERROR(cudaSetDevice(i));
		HANDLE_ERROR(cudaFreeHost(GPU_values[i].h_anisotropy));
		HANDLE_ERROR(cudaFreeHost(GPU_values[i].h_azimuth));
		HANDLE_ERROR(cudaFreeHost(GPU_values[i].h_data));
		HANDLE_ERROR(cudaFreeHost(GPU_values[i].h_angle));


		HANDLE_ERROR(cudaFree(GPU_values[i].d_anisotropy));
		HANDLE_ERROR(cudaFree(GPU_values[i].d_azimuth));
		HANDLE_ERROR(cudaFree(GPU_values[i].d_data));
		
		HANDLE_ERROR(cudaStreamDestroy(GPU_values[i].stream));
		cudaDeviceReset();
	}

	fclose(datTxt);
	fclose(datTxt1);
	fclose(inpCheck);
	fclose(outputAnisotropy00);
	fclose(outputAnisotropy09);
	fclose(outputAnisotropy49);
	fclose(outputAnisotropy99);

	fclose(outputAzimuth00);
	fclose(outputAzimuth09);
	fclose(outputAzimuth49);
	fclose(outputAzimuth99);
	
//	free(data);
	
	size_t free_byte ;

	size_t total_byte ;

	cudaMemGetInfo( &free_byte, &total_byte );
	double free_db = (double)free_byte;
	double total_db = (double)total_byte;
	double used_db = total_db - free_db;
	printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
	return 0;
}
		
