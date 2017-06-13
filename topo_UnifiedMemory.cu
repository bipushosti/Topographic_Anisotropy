



#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <ctype.h>
#include <getopt.h>
#include <string.h>

#include <stdio.h>
#include <cmath>
#include <float.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/device_vector.h>

using namespace std;

//Size of the GPU memory
#define GPU_MEMSIZE_GB		2

//For case in which XSIZE = 1201 and YSIZE = 801
#define GLOBAL_MEM_USE_MB	773
#define MEM_USE_PER_THREAD_B	1280

//MAX_XSIZE_POSSIBLE is the maximum size of x or max number of columns if there is only one row
#define MAX_XSIZE_POSSIBLE	floor(((GPU_MEMSIZE_GB * 1000 - GLOBAL_MEM_USE_MB)*1000000)/MEM_USE_PER_THREAD_B) 

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


//Always have even number of radius;and divisible by 10

#define	RADSTEP			1
#define ANGLESIZE		36	

#define PI 3.14

#define THREADS_PER_BLOCK	512

//#define FILENAME	"Annie_coastDEM.txt"
//---------------------------Function declarations--------------------------------------------------------------------------//

__global__ void getMatrix(float* data,float* angle,float* anisotropy,float* azimuth,float* variance,float* orientation,float* ortho,size_t XSIZE,size_t YSIZE,int RADIUS,int WINDOW_SIZE);
int Get_GPU_devices(void);
static void HandleError( cudaError_t err,const char *file, int line );
//--------------------------------------------------------------------------------------------------------------------------//

//Current Usage:
//Global Memory: 773 MB


__global__ void getMatrix(float* data,float* angle,float* anisotropy,float* azimuth,float* variance,float* orientation,float* ortho,size_t XSIZE,size_t YSIZE,int RADIUS,int WINDOW_SIZE)
//__global__ void getMatrix(int* data,float* angle,float* anisotropy,float* azimuth,size_t XSIZE,size_t YSIZE)
{


//	Thread indices
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int id = y * XSIZE + x;
//----------------------------------------------------------------------------------------------------------------------------//	

	if((y>(YSIZE - RADIUS - 1))||(y<(RADIUS))) return;
	else if((x>(XSIZE - RADIUS - 1))||(x<(RADIUS))) return;
	else
	{

		//Actual computation
		int xrad,yrad,xradOrtho1,yradOrtho1,xradOneEighty,yradOneEighty,valueOneEighty;
		int valueOrtho1,valueOrtho2,xradOrtho2,yradOrtho2,i,j,k;
		//printf("Radius is: %d\n",RADIUS);
	
/*
		float variance[RADIUS];
		float orientation[RADIUS];
		float ortho[RADIUS];
*/		
		

		float value,sum_value,avg_value;
		float sum_valueOrtho,avg_valueOrtho;
		
	//	Initializing declared variables
		sum_value = 0;
		avg_value = 0;
		sum_valueOrtho = 0;
		avg_valueOrtho = 0;

	//	Iniitalize variance, ortho, and orientation arrays with max float value SGR changed i<100 to i<RADIUS

		
				
		//Flipped
		for(i=0;i<ANGLESIZE;i++) {
			
			for(k = 0;k<RADIUS;k+=RADSTEP) {
	
				//Initializing to 0 so that the sum is zero everytime it starts
				sum_value = 0;
				sum_valueOrtho = 0;

				for(j=k;j<k+WINDOW_SIZE;j++){

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
							else {
								avg_value = 1;
							}
					}

					if(avg_valueOrtho == 0) {
						avg_valueOrtho = 1;
					}
				
					//Determine if the variance is minimum compared to  others at scale j, if so record it and its angle i. If not, pass it
					if(avg_value < variance[id * RADIUS + j]) {
							variance[id * RADIUS + j] = avg_value;
							orientation[id * RADIUS + j] = angle[i];
							ortho[id * RADIUS + j] = avg_valueOrtho;		
					}	
				}
			}
		}
		for(j=0;j<RADIUS;j+=RADSTEP){	
			anisotropy[y * XSIZE  * RADIUS/RADSTEP + x * RADIUS/RADSTEP + j] = (36+ortho[id * RADIUS + j])/(36+variance[id * RADIUS + j]);
			azimuth[y * XSIZE  * RADIUS/RADSTEP + x * RADIUS/RADSTEP + j] = orientation[id * RADIUS + j] * 180/PI;
		}
	}
 
}

//--------------------------------------END OF KERNEL-----------------------------------------------------------//

//--------------------------------------Handle Error()-----------------------------------------------------------//

static void HandleError( cudaError_t err,const char *file, int line ) {
    if (err != cudaSuccess) {
		cout << cudaGetErrorString(err) << "in" << file << "at line" << line << "\n";
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
		cout<< "Device does not handle overlaps so streams are not possible\n";
	return 0;
	}

	DeviceCount = 0;
	
	HANDLE_ERROR(cudaGetDeviceCount(&DeviceCount));
	if(DeviceCount > 0){ 
		cout<<  DeviceCount <<"Devices Found\n";
	}else{
		cout<< "No devices found or error in reading the number of devices\n";
		return 0;
	}
	
	for(int i = 0;i<DeviceCount;i++){
		cudaDeviceProp properties;
		HANDLE_ERROR(cudaGetDeviceProperties(&properties,i));
		cout<<"Device Number:"<< i << "\n";
		cout<<"  Device name: "<< properties.name;
		cout<<"  Device Global Memory size: "<< properties.totalGlobalMem/1000000 << "MB \n";
		cout<<"\n";

	}

	return DeviceCount;
}

//-------------------------------------------------------------------------------------------------------------//

int main(int argc,char* argv[])
{

	char FileName[20];
	char delimiterStr[10];
	char delimiter;
	int RADIUS;
	int WINDOW_SIZE;	

	//delimiter_string = "A";
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


	printf("Radius is %d\n",RADIUS);
	printf("Input file name is: %s\n",FileName);

//-------------------------------------------------------------------------------------//

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
	

	datTxt1 = fopen(FileName,"r");	
	if(datTxt1 == NULL) {
		cout<< "Cannot open file:" << argv[1] <<  "\nCheck if file exists.\n";
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
	size_t XSIZE,YSIZE;
	XSIZE = 0;
	YSIZE = 0;
	int i,j;

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
	cout<< "(XSIZE,YSIZE)::"<< "(" << XSIZE << "," << YSIZE << ")" << "\n";

	datTxt = fopen(FileName,"r");

	if(datTxt == NULL) {
		printf("Cannot open file: %s\nCheck if file exists\n",argv[1]);
		exit(1);
	}

//-----------------------Checking if the data size fits the memory of the GPU----------------------------------------------------------------------------------------//
	cout<< "(XSIZE,YSIZE)::"<< "(" << XSIZE << "," << YSIZE << ")" << "\n";
	//(MAX_XSIZE_POSSIBLE - XSIZE*YSIZE >0)? printf("There is enough memory for the computation\n"):printf("There is not enough memory and may result in incorrect results\n");

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------//
//	Allocating Managed Memory (Unified Memory)

//	dim3 gridSize(XSIZE ,(YSIZE+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,1);
//	dim3 blockSize(1,THREADS_PER_BLOCK,1);

	long int total_threads;

	float* data;
	float* anisotropy,*azimuth,*angle;
	float* variance,*orientation,*ortho;

	total_threads = THREADS_PER_BLOCK * ((XSIZE + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK) * YSIZE;

	HANDLE_ERROR(cudaMallocManaged((void**)&angle,ANGLESIZE * sizeof(float)));
	HANDLE_ERROR(cudaMallocManaged((void**)&data,XSIZE * YSIZE * sizeof(float)));
	HANDLE_ERROR(cudaMallocManaged((void**)&anisotropy,YSIZE  * XSIZE  * RADIUS/RADSTEP * sizeof(float)));
	HANDLE_ERROR(cudaMallocManaged((void**)&azimuth,YSIZE  * XSIZE  * RADIUS/RADSTEP * sizeof(float)));

	HANDLE_ERROR(cudaMallocManaged((void**)&variance,total_threads * RADIUS * sizeof(float)));
	HANDLE_ERROR(cudaMallocManaged((void**)&orientation,total_threads * RADIUS * sizeof(float)));
	HANDLE_ERROR(cudaMallocManaged((void**)&ortho,total_threads * RADIUS * sizeof(float)));
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	//XSIZE ints in a row which are max of 5 digits
	//with a space in the front and the back and space
	//between each number 
	char *startPtr,*endPtr;
	char line[XSIZE * 10 +2+(XSIZE-1)];
	memset(line, '\0', sizeof(line));
	int Value;
	i = 0;
	j = 0;
	//Assuming each number in the data set has a max of 5 characters
	char tempVal[5];
	memset(tempVal,'\0',sizeof(tempVal));

	cout<< "Working1\n";
	while(fgets(line,XSIZE *10 + 2 + (XSIZE-1),datTxt)!=NULL) {	
		cout << "Working2\n";
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

				endPtr = endPtr + 1;
				startPtr = endPtr;
			}	
			else if(i == (XSIZE - 1)){
				strcpy(tempVal,startPtr);
				Value = atoi(tempVal);
				data[j * XSIZE + i] = Value;
				fprintf(inpCheck,"%d\n",Value);
			}
		}
		
		j++;
	}	
	
	
//------------------------------------Matrix Declarations--------------------------------------------------------------------------------------------------------------//
//	float angle[ANGLESIZE];

	for(int i=0;i<ANGLESIZE;i++) {
		angle[i] = i * 5 * PI/180;
	}
	
	for(i=0;i<RADIUS * total_threads ;i++){
			variance[i] = FLT_MAX;
			ortho[i] = FLT_MAX;
			orientation[i] = FLT_MAX;
	}
//--------------------------------------CUDA-------------------------------------------------------------------------------------------------------------------------//

	
	cudaError_t error;
	//error = cudaSetDevice(Get_GPU_devices() -1);
	error = cudaSetDevice(0);

	if(error == cudaSuccess){
		 cout <<"success\n";
	}else{
		cout <<"unsuccessful\n";
	}
	
	//cudaSetDevice(1);

	cout<< "Hello1\n";

	dim3 gridSize((XSIZE + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK,YSIZE,1);
	dim3 blockSize(THREADS_PER_BLOCK,1,1);

	//dim3 gridSize(3,YSIZE,1);
	

	cout <<"Hello2\n";

	getMatrix<<<gridSize,blockSize>>>(data,angle,anisotropy,azimuth,variance,orientation,ortho,XSIZE,YSIZE,RADIUS,WINDOW_SIZE);

	error = cudaDeviceSynchronize();
	if(error != cudaSuccess){
		cout << "CUDA Device Synchronization Error:" << cudaGetErrorString(error) << "\n";

    		// we can't recover from the error -- exit the program
    		return 0;
  	}

	error = cudaGetLastError();

	if(error != cudaSuccess){
		cout <<"CUDA Error:" << cudaGetErrorString(error) << "\n";
    		// we can't recover from the error -- exit the program
	    	return 0;
  	}

	cout << "Hello3\n";
	cout << "Hello4\n";
	cout << "Hello5\n";

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------//
//			Writing to files


	for(j=0;j<YSIZE ;j++) {
		for(i=0;i<XSIZE ;i++) {
			if((j>(YSIZE - RADIUS - 1))||(j<(RADIUS))) continue;
			if((i>(XSIZE - RADIUS - 1))||(i<(RADIUS))) continue;

			if (i == (XSIZE  - RADIUS - 1)) {
				fprintf(outputAnisotropy00,"%f",anisotropy[j * XSIZE  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + 0]);
				fprintf(outputAzimuth00,"%f",azimuth[j * XSIZE * RADIUS/RADSTEP + i * RADIUS/RADSTEP + 0]);
				fprintf(outputAnisotropy00,"\n");
				fprintf(outputAzimuth00,"\n");

				fprintf(outputAnisotropy09,"%f",anisotropy[j * XSIZE  * RADIUS/RADSTEP + i * RADIUS/RADSTEP +RADIUS/10 -1]);
				fprintf(outputAzimuth09,"%f",azimuth[j * XSIZE  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + RADIUS/10 -1]);
				fprintf(outputAnisotropy09,"\n");
				fprintf(outputAzimuth09,"\n");

				fprintf(outputAnisotropy49,"%f",anisotropy[j * XSIZE  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + RADIUS/2 - 1]);
				fprintf(outputAzimuth49,"%f",azimuth[j * XSIZE  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + RADIUS/2 - 1]);
				fprintf(outputAnisotropy49,"\n");
				fprintf(outputAzimuth49,"\n");

				fprintf(outputAnisotropy99,"%f",anisotropy[j * XSIZE  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + RADIUS -1]);
				fprintf(outputAzimuth99,"%f",azimuth[j * XSIZE  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + RADIUS - 1]);
				fprintf(outputAnisotropy99,"\n");
				fprintf(outputAzimuth99,"\n");
			}
			else {
				fprintf(outputAnisotropy00,"%f",anisotropy[j * XSIZE  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + 0]);
				fprintf(outputAzimuth00,"%f",azimuth[j * XSIZE  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + 0]);
				fprintf(outputAnisotropy00,"\t");
				fprintf(outputAzimuth00,"\t");
	
				fprintf(outputAnisotropy09,"%f",anisotropy[j * XSIZE  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + RADIUS/10 -1]);
				fprintf(outputAzimuth09,"%f",azimuth[j * XSIZE  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + RADIUS/10 -1]);
				fprintf(outputAnisotropy09,"\t");
				fprintf(outputAzimuth09,"\t");

				fprintf(outputAnisotropy49,"%f",anisotropy[j * XSIZE  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + RADIUS/2 - 1]);
				fprintf(outputAzimuth49,"%f",azimuth[j * XSIZE  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + RADIUS/2 - 1]);	
				fprintf(outputAnisotropy49,"\t");
				fprintf(outputAzimuth49,"\t");

				fprintf(outputAnisotropy99,"%f",anisotropy[j * XSIZE  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + RADIUS - 1]);
				fprintf(outputAzimuth99,"%f",azimuth[j * XSIZE  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + RADIUS - 1]);
				fprintf(outputAnisotropy99,"\t");
				fprintf(outputAzimuth99,"\t");	
			}					
		}
	}	


	fclose(datTxt);
	fclose(inpCheck);
	fclose(outputAnisotropy00);
	fclose(outputAnisotropy09);
	fclose(outputAnisotropy49);
	fclose(outputAnisotropy99);

	fclose(outputAzimuth00);
	fclose(outputAzimuth09);
	fclose(outputAzimuth49);
	fclose(outputAzimuth99);
	
	cudaFree(data);
	cudaFree(angle);
	cudaFree(azimuth);
	cudaFree(anisotropy);
	cudaFree(variance);
	cudaFree(orientation);
	cudaFree(ortho);
	
	
	//free(max_line);
//	free(anisotropy);
//	free(azimuth);

//	size_t free_byte ;

//	size_t total_byte ;
/*
	cudaMemGetInfo( &free_byte, &total_byte );
	double free_db = (double)free_byte;
	double total_db = (double)total_byte;
	double used_db = total_db - free_db;
//	cout << "GPU memory usage: used = %f, free = %f MB, total = %f MB\n",used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
*/
	return 0;
}
