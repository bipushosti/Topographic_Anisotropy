//Input file: space delimited

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
//#include <helper_functions.h>
//#include <helper_cuda.h>
#include <assert.h>
#include "topographic_anisotropy_largerGrid.h"

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
#define RADIUS			100

#define	RADSTEP			1
#define ANGLESIZE		36	//Size of angle array	

#define PI 			3.14159


#define THREADS_PER_BLOCK	512

//__constant__ int RADIUS;

//#define FILENAME	"Annie_coastDEM.txt"
//---------------------------Function and Global variable declarations--------------------------------------------------------------------------//

__global__ void getMatrix(int* data,float* angle,float* anisotropy,float* azimuth,long int XSIZE,long int YSIZE);
int Get_GPU_devices();
static void HandleError(cudaError_t err,const char *file, int line);
inline cudaError_t checkCuda(cudaError_t result);


//--------------------------------------------------------------------------------------------------------------------------//

__global__ void getMatrix(int* data,float* angle,float* anisotropy,float* azimuth,long int XSIZE,long int YSIZE)
{

	//The kernel does not use the new definition of RADIUS in main but the one at the top of the file
	//Therefore the define at the top and the input value of RADIUS must be equal (For now)

	//printf("The RADIUS is: %d\n",RADIUS);
	

//	Thread indices
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
//----------------------------------------------------------------------------------------------------------------------------//	

	//y>YSIZE - 1 for multi-gpu code because the computation needs to be done upto the radius
	if((y>(YSIZE - 1))||(y<(RADIUS))) return;
	else if((x>(XSIZE - RADIUS - 1))||(x<(RADIUS))) return;
	else
	{
		//printf("%d,%d\n",y,x);
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

	char delimiter;

	//delimiter_string = "A";
	if(argc == 1){
		printf("\tNot enough arguments\n");
		printf("\t\tUsage: ./Executable DataFileName DataFileDelimiter Radius\n");
		return 0;
	}

	//In the future use optarg
	if(strcmp(argv[2],"space")==0){
		delimiter = ' ';
	}
	else if(strcmp(argv[2],"Space")==0){
		delimiter = ' ';
	}else{
		delimiter = *argv[2];
	}
	
	printf("Delimiter: %c\n",delimiter);
	//return 0;

	#undef RADIUS
	#define RADIUS atoi(argv[3])

	//RADIUS = tmp;
	printf("Radius is %d\n",RADIUS);


//-------------------------------------------------------------------------------------//
	//Setting the output buffer to 500MB
	size_t limit;
	HANDLE_ERROR(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 500 * 1024 * 1024));
	cudaDeviceGetLimit(&limit,cudaLimitPrintfFifoSize);
	
	//Setting the heap size 
	HANDLE_ERROR(cudaDeviceSetLimit(cudaLimitMallocHeapSize,10 * 1000 * 1000 * 4));

//-------------------------------------------------------------------------------------//	
	//File declarations and opening them
	FILE *datTxt1,*datTxt;
	FILE *outputAnisotropy00,*outputAnisotropy09,*outputAnisotropy24,*outputAnisotropy49,*outputAnisotropy99;
	FILE *outputAzimuth00,*outputAzimuth09,*outputAzimuth24,*outputAzimuth49,*outputAzimuth99; 
	
	FILE *outputAnisotropy04,*outputAzimuth04;

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

//-------------------------------------------------------------------------------------//
//				Setting Up Output Filenames
//-------------------------------------------------------------------------------------//

	char *lastSlash;
	char FileName[20];
	char AniFirst[80],AniFive[80],AniTen[80],AniTwentyFive[80],AniFifty[80],AniLast[80];
	char AziFirst[80],AziFive[80],AziTen[80],AziTwentyFive[80],AziFifty[80],AziLast[80];

	memset(FileName,'\0',sizeof(FileName));

	lastSlash = strrchr(argv[1],'/');

	if(lastSlash == NULL){
		strcpy(FileName,argv[1]);
	}
	else{
		printf("Found slash at %s\n",lastSlash);
		strcpy(FileName,lastSlash+1);
	}
	printf("FileName is %s\n",FileName);

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

	datTxt = fopen(argv[1],"r");
	if(datTxt == NULL) {
		//printf("Cannot open file: %s\nCheck if file exists\n",argv[1]);
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

	printf("Done data[%zd][%zd] = %f\n",j-1,i-1,*(data + 500 * XSIZE + 500));	
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
	long int tmpSize = 0;
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
		printf("\n########################Device %ld #############################\n",i);

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
		printf("Row Value is: %ld\n",GPU_values[i].NumRows);

		if((i == 0) ||(i == (DeviceCount -1))){
			GPU_values[i].size = (GPU_values[i].NumRows + RADIUS ) * XSIZE;	
			printf("Size is: %ld\n",GPU_values[i].size);
			printf("i is: %ld\n",i);
		//Sections in between
		}else{
			GPU_values[i].size = (GPU_values[i].NumRows + 2*RADIUS) * XSIZE;
			//offset = RADIUS * -1;
		}
		printf("Size is: (GPU_values[%zd].NumRows + RADIUS) * XSIZE *sizeof(int) = (%ld + %d )*%ld *%ld =  %ld\n",i,GPU_values[i].NumRows,RADIUS,XSIZE,sizeof(float),GPU_values[i].size*sizeof(float));	
	}

	//return 0;

	for(i = 0;i<DeviceCount;i++){

		printf("\n########################Device %ld #############################\n",i);
		printf("Radius is %d\n",RADIUS);
		//-----------------Matrix Allocations----------------------------//
		HANDLE_ERROR(cudaSetDevice(i));
		HANDLE_ERROR(cudaStreamCreate(&GPU_values[i].stream));
		HANDLE_ERROR(cudaDeviceSetLimit(cudaLimitMallocHeapSize, (size_t)(GPU_values[i].size *sizeof(int) + ANGLESIZE * sizeof(float) + 2*GPU_values[i].size * RADIUS/RADSTEP * sizeof(float))));

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

		printf("GridSize(X,Y) = (%ld,%ld)\n",(GPU_values[i].NumCols + THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,(GPU_values[i].NumRows+RADIUS));

		//----------------Launching the Kernel---------------------//
		printf("Radius is %d\n",RADIUS);
		getMatrix<<<gridSize,blockSize,0,GPU_values[i].stream>>>(GPU_values[i].d_data,GPU_values[i].d_angle,GPU_values[i].d_anisotropy,GPU_values[i].d_azimuth,GPU_values[i].NumCols,GPU_values[i].NumRows);

		HANDLE_ERROR(cudaDeviceSynchronize());
		//getLastCudaError("Kernel failed \n");

		

		//---------------Getting data back------------------------//
		HANDLE_ERROR(cudaMemcpyAsync(GPU_values[i].h_anisotropy,GPU_values[i].d_anisotropy,GPU_values[i].size * RADIUS/RADSTEP * sizeof(float),cudaMemcpyDeviceToHost,GPU_values[i].stream));
		HANDLE_ERROR(cudaMemcpyAsync(GPU_values[i].h_azimuth,GPU_values[i].d_azimuth,GPU_values[i].size * RADIUS/RADSTEP * sizeof(float),cudaMemcpyDeviceToHost,GPU_values[i].stream));

		printf("Device # %ld\n",i);
		
	}
	int z;
	for(z = 0;z<DeviceCount;z++){

		HANDLE_ERROR(cudaSetDevice(z));
		HANDLE_ERROR(cudaStreamSynchronize(GPU_values[z].stream));

		printf("Device l%d: Rows: %ld,Cols: %ld\n",z,GPU_values[z].NumRows,GPU_values[z].NumCols);
		printf("Radius is: %d\n",RADIUS);

		for(j=0;j<GPU_values[z].NumRows ;j++) {

			for(i=0;i<GPU_values[z].NumCols ;i++) {

				if((j>(GPU_values[z].NumRows - 1))||(j<(RADIUS))) continue;
				if((i>(GPU_values[z].NumCols - RADIUS - 1))||(i<(RADIUS))) continue;

				//printf("Col:%ld,Row: %ld\n",i,j);
				if (i == (GPU_values[z].NumCols  - RADIUS - 1)) {
					fprintf(outputAnisotropy00,"%f",GPU_values[z].h_anisotropy[j * GPU_values[z].NumCols  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + 0]);
					fprintf(outputAzimuth00,"%f",GPU_values[z].h_azimuth[j * GPU_values[z].NumCols * RADIUS/RADSTEP + i * RADIUS/RADSTEP + 0]);
					fprintf(outputAnisotropy00,"\n");
					fprintf(outputAzimuth00,"\n");

					fprintf(outputAnisotropy04,"%f",GPU_values[z].h_anisotropy[j * GPU_values[z].NumCols  * RADIUS/RADSTEP + i * RADIUS/RADSTEP + 5]);
					fprintf(outputAzimuth04,"%f",GPU_values[z].h_azimuth[j * GPU_values[z].NumCols * RADIUS/RADSTEP + i * RADIUS/RADSTEP + 5]);
					fprintf(outputAnisotropy04,"\n");
					fprintf(outputAzimuth04,"\n");

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
		
