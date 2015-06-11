



//Input file: space delimited

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <cuda.h>
#include <cuda_runtime.h>

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

#define PI 3.14


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
	
//-----------Getting the number of devices and their sizes---------------------------------------------------------------------------------------------------//
	int DeviceCount;	
	DeviceCount = Get_GPU_devices();

//----------Getting number of rows for each GPU-------------------------------------------------------------------------------------------------------------//

	int tmpSize = 0;
	int count = 0;
	//GPU_sizes holds the number of rows each of the GPU devices will hold 
	int GPU_sizes[DeviceCount];
	tmpSize = YSIZE;
	count = DeviceCount;

	int* data_ptr[DeviceCount];
	float *anisotropy_ptr[DeviceCount],*azimuth_ptr[DeviceCount],*angle_ptr;
	
	for(int i = 0;i<DeviceCount;i++){
		if(tmpSize % count != 0){
			GPU_sizes[i] = (tmpSize/count) + 1;
		}else{
			GPU_sizes[i] = tmpSize/count;
		}	
		tmpSize = tmpSize - GPU_sizes[i];
		count--;
		printf("%d\n",GPU_sizes[i]);
	}

//------------------------------------Matrix Declarations-----------------------------------------------------------------------------------------------------//
	float angle[ANGLESIZE];
	for(int i=0;i<ANGLESIZE;i++) {
		angle[i] = i * 5 * PI/180;
		//printf("%d	::	%f\n",i,angle[i]);
	}

	float* anisotropy[DeviceCount],*azimuth[DeviceCount];

	int dev = 0;
	int pos = 0;

	//Allocating memory according to the section of the data
	for(dev = 0; dev<DeviceCount; dev++){
		if((dev == 0) ||(dev == (DeviceCount -1))){
			anisotropy[dev] = (float*)malloc((GPU_sizes[dev]+RADIUS)  * XSIZE  * RADIUS/RADSTEP * sizeof(float));
			azimuth[dev] = (float*)malloc((GPU_sizes[dev]+RADIUS)  * XSIZE  * RADIUS/RADSTEP * sizeof(float));
		}else{
			anisotropy[dev] = (float*)malloc((GPU_sizes[dev]+2*RADIUS)  * XSIZE  * RADIUS/RADSTEP * sizeof(float));
			azimuth[dev] = (float*)malloc((GPU_sizes[dev]+2*RADIUS)  * XSIZE  * RADIUS/RADSTEP * sizeof(float));
		}
	}

//----------Dividing and allocating memory for the data in GPU and sending data-----------------------------------------------------------------------------------------//

	//size holds the total size of the data for each of the parts
	size_t size = 0;
	//offset holds either 2*RADIUS or RADIUS depending on the part of data
	int offset = 0;

	for(dev = 0,pos=0;dev<DeviceCount; pos+=GPU_sizes[dev],dev++){

		cudaSetDevice(dev);
		//If the first section or last
		if((dev == 0) ||(dev == (DeviceCount -1))){
			size = (GPU_sizes[dev] + RADIUS) * XSIZE * sizeof(float);
			
		//Sections in between
		}else{
			size = (GPU_sizes[dev] + 2*RADIUS) * XSIZE * sizeof(float);
		}

		//Setting the offset into the data matrix
		if(dev == 0){
			offset = 0;
		}else{
			offset = RADIUS * -1;
		}



		//cudaMalloc((void**)&data_ptr[dev],GPU_sizes[dev] * XSIZE * sizeof(float));
		HANDLE_ERROR(cudaMalloc((void**)&data_ptr[dev],size));
		HANDLE_ERROR(cudaMalloc((void**)&angle_ptr,ANGLESIZE * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&anisotropy_ptr[dev],size * RADIUS/RADSTEP));
		HANDLE_ERROR(cudaMalloc((void**)&azimuth_ptr[dev],size  * RADIUS/RADSTEP));

//CHECKED!!----Getting the correct values from the data array	
		//for (int j = 0;j<size/sizeof(float);j++){
		//	printf("%d\n",*(data + XSIZE* (pos+offset)+j));
		//}
		//printf("----------------------------------------------------------------\n");
		//printf("Size: %zd\n",size/sizeof(float));		

		//return 0;

//ERROR HERE!!
		HANDLE_ERROR(cudaMemcpy(data_ptr[dev],(pos+offset) * XSIZE + data,size,cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(angle_ptr,angle,ANGLESIZE * sizeof(float),cudaMemcpyHostToDevice));
	}

/*
//----------Sending data to the GPU-----------------------------------------------------------------------------------------------------------------------------//
	//int pos = 0;
	for(dev = 0,pos=0;dev<DeviceCount; pos+=GPU_sizes[dev],dev++){
		cudaSetDevice(dev);
		if((dev == 0) ||(dev == (DeviceCount -1))){
			size = (GPU_sizes[dev] + RADIUS) * XSIZE * sizeof(float);
		}else{
			size = (GPU_sizes[dev] + 2*RADIUS) * XSIZE * sizeof(float);
		}
		cudaMemcpy(data_ptr[dev],pos * XSIZE + data,size,cudaMemcpyHostToDevice);
		cudaMemcpy(angle_ptr,angle,ANGLESIZE * sizeof(float),cudaMemcpyHostToDevice);

		//cudaMemcpy(anisotropy_ptr[dev],pos * XSIZE * RADIUS/RADSTEP + data,GPU_sizes[dev] * XSIZE * RADIUS/RADSTEP * sizeof(float),cudaMemcpyHostToDevice);
		//cudaMemcpy(azimuth_ptr[dev],pos * XSIZE * RADIUS/RADSTEP + data,GPU_sizes[dev] * XSIZE * RADIUS/RADSTEP * sizeof(float),cudaMemcpyHostToDevice);
	}
*/

//--------------------------------------CUDA--------------------------------------------------------------------------------------------------------------------//
	//dim3 gridSize((XSIZE + THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK ,YSIZE,1);
	dim3 blockSize(THREADS_PER_BLOCK,1,1);

	printf("Hello2\n");
	int rows_y = 0;



	//for(int i = 0;i <251*XSIZE;i++){
	//	printf("%d\n",data[i]);
	//}



	for(dev = 0; dev < DeviceCount; dev++){

		cudaSetDevice(dev);
		if((dev == 0) ||(dev == (DeviceCount -1))){
			rows_y = GPU_sizes[dev] + RADIUS;
		//Sections in between
		}else{
			rows_y = GPU_sizes[dev] + 2 * RADIUS;
		}
		dim3 gridSize((XSIZE + THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK ,rows_y,1);
		getMatrix<<<gridSize,blockSize>>>(data_ptr[dev],angle_ptr,anisotropy_ptr[dev],azimuth_ptr[dev],XSIZE,rows_y);
	}

	printf("Hello3\n");
	//cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
  	{
		printf("Error\n");
		printf("CUDA Error: %s\n", cudaGetErrorString(error));

    	// we can't recover from the error -- exit the program
    	return 0;
  	}

	
	rows_y = 0;	
	for(dev = 0; dev < DeviceCount;dev++){
		cudaSetDevice(dev);
		if((dev == 0) ||(dev == (DeviceCount -1))){
			rows_y = GPU_sizes[dev] + RADIUS;
		//Sections in between
		}else{
			rows_y = GPU_sizes[dev] + 2 * RADIUS;
		}

//For error checking only 
/*
		HANDLE_ERROR(cudaMalloc((void**)&anisotropy_ptr[dev],size * RADIUS/RADSTEP));
		HANDLE_ERROR(cudaMalloc((void**)&azimuth_ptr[dev],size  * RADIUS/RADSTEP));
*/
		HANDLE_ERROR(cudaMemcpy(anisotropy[dev],anisotropy_ptr[dev],rows_y  * XSIZE  * RADIUS/RADSTEP * sizeof(float),cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(azimuth[dev],azimuth_ptr[dev],rows_y  * XSIZE  * RADIUS/RADSTEP * sizeof(float),cudaMemcpyDeviceToHost));
	}

	printf("Hello4\n");

	for(dev = 0; dev<DeviceCount; dev++){

		cudaFree(data_ptr[i]);	
		
		cudaFree(azimuth_ptr[i]);
		cudaFree(anisotropy_ptr[i]);
	}
	cudaFree(angle_ptr);

	printf("%f\n",*(anisotropy[0]+1));
	printf("Hello5\n");

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------//
//			Writing to files
/*

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

*/
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
	

	
	//free(max_line);
	for(dev = 0; dev<DeviceCount;dev++){
		free(anisotropy[dev]);
		free(azimuth[dev]);
	}
	
	size_t free_byte ;

	size_t total_byte ;

	cudaMemGetInfo( &free_byte, &total_byte );
	double free_db = (double)free_byte;
	double total_db = (double)total_byte;
	double used_db = total_db - free_db;
	printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
	return 0;
}
