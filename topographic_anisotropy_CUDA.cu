

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <cuda.h>

#define XSIZE 	1201
#define YSIZE	801



#define RADIUS		100
#define	RADSTEP		1
#define ANGLESIZE	72	


#define PI 3.14

//---------------------------Function declarations--------------------------------------------------------------------------//

__global__ void getMatrix(int* data,float* cmatrix,float* cor,float* cor_bi,float* angle,float* anisotropy,float* azimuth);

//--------------------------------------------------------------------------------------------------------------------------//



__global__ void getMatrix(int* data,float* cmatrix,float* cor,float* cor_bi,float* angle,float* anisotropy,float* azimuth)
{

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
//	int index = col + row * RADIUS/RADSTEP;

	//int x = col + 100;
	//int y = row + 100;
	if((y<701)&&(y>100) && (x<1101)&&(x> 100)) {

		int xrad,yrad,i,j,k,index1,cor_bi_MinInd;
		xrad = 0;
		yrad = 0;

		//float cmatrix[ANGLESIZE][RADIUS/RADSTEP];
		//float cor[ANGLESIZE][RADIUS/RADSTEP];
		//float cor_bi[ANGLESIZE/2][RADIUS/RADSTEP];


		float tempSum,tempCompute,cor_bi_ColMin,cor_bi_Ortho;

		for(j = 0;j<RADIUS;j+=RADSTEP) {
			for(i=0;i<ANGLESIZE;i++) {
				xrad = (int)round(cosf(angle[i]) * (j+1) + x);	
				yrad = (int)round(sinf(angle[i]) * (j+1) + y);	

				cmatrix[i * RADIUS/RADSTEP + j] = (float)data[(yrad-1 ) * XSIZE + xrad-1]; 	

				tempSum = 0;
				tempCompute = 0;

				for(index1 = 0;index1<=j;index1++) {					
					tempCompute = cmatrix[i * RADIUS/RADSTEP + index1] - (float)data[(y-1) * XSIZE + x-1];
					tempCompute  = tempCompute * tempCompute ;
					tempSum = (tempSum + tempCompute);
				}
		
				cor[i * RADIUS/RADSTEP + j] = tempSum/(2*(j+1));	
			}
	
			cor_bi_ColMin = FLT_MAX;
			cor_bi_MinInd = 0;
			cor_bi_Ortho = 0;
			for (k=0;k<(ANGLESIZE)/2;k++) {

				cor_bi[k * RADIUS/RADSTEP + j] = (cor[k * RADIUS/RADSTEP + j] + cor[(k+36) * RADIUS/RADSTEP + j])/2 ;

				if(cor_bi[k *RADIUS/RADSTEP + j] < cor_bi_ColMin) {					
					cor_bi_ColMin = cor_bi[k *RADIUS/RADSTEP + j];
					cor_bi_MinInd = k;
				}
			}

			if(cor_bi_MinInd <18) {								
				cor_bi_Ortho = cor_bi[(cor_bi_MinInd + 18)* RADIUS/RADSTEP + j];
			}
			else {
				cor_bi_Ortho = cor_bi[(cor_bi_MinInd - 18)*RADIUS/RADSTEP + j];
			}		

			//Fail safe to ensure there is no nan or inf			
			if(cor_bi_ColMin == 0) {
					if((cor_bi_Ortho < 1) && (cor_bi_Ortho > 0)) {
						cor_bi_ColMin = cor_bi_Ortho;
					}
					else {
						cor_bi_ColMin = 1;
					}
			}

			if(cor_bi_Ortho == 0) {
				cor_bi_Ortho = 1;
			}
		}
		
		anisotropy[y * YSIZE * RADIUS/RADSTEP + x * RADIUS/RADSTEP + j] = cor_bi_Ortho/cor_bi_ColMin;
		azimuth[y * YSIZE * RADIUS/RADSTEP + x * RADIUS/RADSTEP + j] = angle[cor_bi_MinInd] * 180/PI ;	
		
	}
 
}

//--------------------------------------END OF KERNEL-----------------------------------------------------------//




int main()
{
	FILE *datTxt,*outputAnisotropy00,*outputAnisotropy09,*outputAnisotropy49,*outputAnisotropy99;
	FILE *outputAzimuth00,*outputAzimuth09,*outputAzimuth49,*outputAzimuth99; 
	int data[YSIZE][XSIZE];

	FILE * inpCheck;
	inpCheck = fopen("inpCheck.txt","w");
	if(inpCheck == NULL) {
		perror("Cannot open dat.txt file");
		return (-1);
	}
	//1200 ints in a row which are max of 5 digits
	//with a space in the front and the back and space
	//between each number 
	char line[1200 * 5 +2+1200];
	memset(line, '\0', sizeof(line));
	char *startPtr,*endPtr;
	
	datTxt = fopen("dat.txt","r");
	if(datTxt == NULL) {
		perror("Cannot open dat.txt file");
		return (-1);
	}

	outputAnisotropy00 = fopen("outputDataAni00.txt","w");
	outputAnisotropy09 = fopen("outputDataAni09.txt","w");
	outputAnisotropy49 = fopen("outputDataAni49.txt","w");
	outputAnisotropy99 = fopen("outputDataAni99.txt","w");
	if((outputAnisotropy00 == NULL)||(outputAnisotropy09 == NULL)||(outputAnisotropy49 == NULL)||(outputAnisotropy99 == NULL)) {
		perror("Cannot open Anisotropy file");
		return (-1);
	}

	outputAzimuth00 = fopen("outputDataAzi00.txt","w");
	outputAzimuth09 = fopen("outputDataAzi09.txt","w");
	outputAzimuth49 = fopen("outputDataAzi49.txt","w");
	outputAzimuth99 = fopen("outputDataAzi99.txt","w");

	if((outputAzimuth00 == NULL)||(outputAzimuth09 == NULL)||(outputAzimuth49 == NULL)||(outputAzimuth99 == NULL)) {
		perror("Cannot open Azimuth file");
		return (-1);
	}

	int i,j,Value;
	j = 0;
	char tempVal[5];
	memset(tempVal,'\0',sizeof(tempVal));

	while(fgets(line,1200 *5 + 2 + 1200,datTxt)!=NULL) {	
		startPtr = line;	
		for(i=0;i<XSIZE;i++) {
			Value = 0;
			memset(tempVal,'\0',sizeof(tempVal));		
			if(i != (XSIZE - 1)) {	
				endPtr = strchr(startPtr,' ');
				strncpy(tempVal,startPtr,endPtr-startPtr); 
				Value = atoi(tempVal);
				data[j][i] = Value;
				fprintf(inpCheck,"%d ",Value);

				endPtr = endPtr + 1;
				startPtr = endPtr;
			}	
			else if(i == (XSIZE - 1)){
				strcpy(tempVal,startPtr);
				Value = atoi(tempVal);
				data[j][i] = Value;
				fprintf(inpCheck,"%d\n",Value);
			}
		}
		
		j++;
	}	
	//return 0;

	float angle[ANGLESIZE];
	for(int i=0;i<ANGLESIZE;i++) {
		angle[i] = i * 5 * PI/180;
		//printf("%d	::	%f\n",i,angle[i]);
	}
	

	//Initializing 2D cmatrix
	float** cmatrix;
	cmatrix = (float**)malloc(ANGLESIZE * sizeof(float*));
	for(i=0;i<ANGLESIZE;i++) {
		cmatrix[i] = (float*)malloc(RADIUS/RADSTEP *sizeof(float));
	}

	//Initializing cor
	float** cor;
	cor = (float**)malloc(ANGLESIZE * sizeof(float*));
	for(i=0;i<ANGLESIZE;i++) {
		cor[i] = (float*)malloc(RADIUS/RADSTEP *sizeof(float));
	}

	//Initializing cor_bi
	float** cor_bi;
	cor_bi = (float**)malloc(ANGLESIZE/2 * sizeof(float*));
	for(i=0;i<ANGLESIZE/2;i++) {
		cor_bi[i] = (float*)malloc(RADIUS/RADSTEP *sizeof(float));
	}

	//Initializing 3D matrix anisotropy
	float*** anisotropy;
	anisotropy = (float***)malloc(YSIZE * sizeof(float**));
	for(i = 0;i<YSIZE;i++) {
		anisotropy[i] = (float**)malloc(XSIZE * sizeof(float *));
		for(j = 0; j<XSIZE;j++) {
			anisotropy[i][j] = (float*)malloc(RADIUS/RADSTEP * sizeof(float));
		}
	}

	//Initializing 3D matrix anzimuth
	float*** azimuth;
	azimuth = (float***)malloc(YSIZE * sizeof(float**));
	for(i = 0;i<YSIZE;i++) {
		azimuth[i] = (float**)malloc(XSIZE * sizeof(float *));
		for(j = 0; j<XSIZE;j++) {
			azimuth[i][j] = (float*)malloc(RADIUS/RADSTEP * sizeof(float));
		}
	}
	
	


	//Actual computation
	int xrad,yrad,x,y,k,index1,cor_bi_MinInd;
	float tempCompute,tempSum,cor_bi_ColMin,cor_bi_Ortho;

//--------------------------------------CUDA-----------------------------------------------------//



	int* ptrD_data,*ptrH_data;
	float* ptrD_angle,*ptrD_cmatrix,*ptrD_cor,*ptrD_corbi,*ptrD_azimuth,*ptrD_anisotropy;
	float *ptrH_anisotropy,*ptrH_azimuth,*ptrH_angle,*ptrH_cor,*ptrH_corbi,*ptrH_cmatrix;
	ptrH_angle = &angle[0];
	ptrH_data = &data[0][0];
	ptrH_cmatrix = &cmatrix[0][0];
	ptrH_cor = &cor[0][0];
	ptrH_corbi = &cor_bi[0][0];
	ptrH_azimuth = &azimuth[0][0][0];
	ptrH_anisotropy = &anisotropy[0][0][0];

	cudaMalloc((void**)&ptrD_data,XSIZE * YSIZE * sizeof(int));
	cudaMalloc((void**)&ptrD_cmatrix,ANGLESIZE * RADIUS/RADSTEP * sizeof(float));
	cudaMalloc((void**)&ptrD_cor,ANGLESIZE * RADIUS/RADSTEP * sizeof(float));
	cudaMalloc((void**)&ptrD_corbi,ANGLESIZE/2 * RADIUS/RADSTEP * sizeof(float));

	cudaMalloc((void**)&ptrD_angle,ANGLESIZE * sizeof(float));
	cudaMalloc((void**)&ptrD_azimuth,YSIZE * XSIZE * RADIUS/RADSTEP * sizeof(float));
	cudaMalloc((void**)&ptrD_anisotropy,YSIZE * XSIZE * RADIUS/RADSTEP * sizeof(float));

	cudaMemcpy(ptrD_data,ptrH_data,XSIZE * YSIZE * sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(ptrD_cmatrix,ptrH_cmatrix,ANGLESIZE * RADIUS/RADSTEP * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(ptrD_cor,ptrH_cor,ANGLESIZE *RADIUS/RADSTEP * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(ptrD_corbi,ptrH_corbi,ANGLESIZE/2 * RADIUS/RADSTEP * sizeof(float),cudaMemcpyHostToDevice);

	cudaMemcpy(ptrD_azimuth,ptrH_azimuth,YSIZE * XSIZE * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(ptrD_anisotropy,ptrH_anisotropy,XSIZE*YSIZE* sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(ptrD_angle,angle,ANGLESIZE * sizeof(float),cudaMemcpyHostToDevice);
	printf("Hello\n");


	const dim3 gridSize(38,YSIZE,1);
	const dim3 blockSize(32,1,1);
	
	getMatrix<<<gridSize,blockSize>>>(ptrD_data,ptrD_cmatrix,ptrD_cor,ptrD_corbi,ptrD_angle,ptrD_anisotropy,ptrD_azimuth);

	printf("Hello2\n");


	cudaMemcpy(ptrH_azimuth,ptrD_azimuth,YSIZE * XSIZE * RADIUS/RADSTEP * sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(ptrH_anisotropy,ptrD_anisotropy,YSIZE * XSIZE * RADIUS/RADSTEP * sizeof(float),cudaMemcpyDeviceToHost);


	printf("Hello3\n");
	
	printf("Hello4\n");
//	cudaFree(ptrH_anisotropy);
//	cudaFree(ptrH_azimuth);

	cudaFree(ptrD_data);
	cudaFree(ptrD_angle);
	cudaFree(ptrD_azimuth);
	cudaFree(ptrD_anisotropy);
	
	cudaFree(ptrD_cmatrix);
	cudaFree(ptrD_cor);
	cudaFree(ptrD_corbi);

//------------------------------------------------------------------------------------------------//
//			Writing to files


/*
	for(j = 0;j<YSIZE;j++) {
			for(i=0;i<XSIZE;i++) {

				if (i == (XSIZE - RADIUS - 1)) {
					fprintf(outputAnisotropy00,"%f\n",anisotropy[i][j][0]);
				}
				else {
					fprintf(outputAnisotropy00,"%f\t",anisotropy[i][j][0]);
				}
			}

	}
*/
	printf("Hello5\n");


	/*if(j == 0) {
		if (x == (XSIZE - RADIUS - 1)) {
			fprintf(outputAnisotropy00,"%f",anisotropy[y][x][j]);
			fprintf(outputAzimuth00,"%f",azimuth[y][x][j]);
			fprintf(outputAnisotropy00,"\n");
			fprintf(outputAzimuth00,"\n");
		}
		else {
			fprintf(outputAnisotropy00,"%f",anisotropy[y][x][j]);
			fprintf(outputAzimuth00,"%f",azimuth[y][x][j]);
			fprintf(outputAnisotropy00,"\t");
			fprintf(outputAzimuth00,"\t");
		}
	}

	else if(j == 9) {
		if (x == (XSIZE - RADIUS - 1)) {
			fprintf(outputAnisotropy09,"%f",anisotropy[y][x][j]);
			fprintf(outputAzimuth09,"%f",azimuth[y][x][j]);
			fprintf(outputAnisotropy09,"\n");
			fprintf(outputAzimuth09,"\n");
		}
		else {
			fprintf(outputAnisotropy09,"%f",anisotropy[y][x][j]);
			fprintf(outputAzimuth09,"%f",azimuth[y][x][j]);
			fprintf(outputAnisotropy09,"\t");
			fprintf(outputAzimuth09,"\t");
		}
	}
	else if(j == 49) {
	
		if (x == (XSIZE - RADIUS - 1)) {
			fprintf(outputAnisotropy49,"%f",anisotropy[y][x][j]);
			fprintf(outputAzimuth49,"%f",azimuth[y][x][j]);
			fprintf(outputAnisotropy49,"\n");
			fprintf(outputAzimuth49,"\n");
		}
		else {
			fprintf(outputAnisotropy49,"%f",anisotropy[y][x][j]);
			fprintf(outputAzimuth49,"%f",azimuth[y][x][j]);	
			fprintf(outputAnisotropy49,"\t");
			fprintf(outputAzimuth49,"\t");
		}
	}
	else if(j == 99) {
	
		if (x == (XSIZE - RADIUS - 1)) {
			fprintf(outputAnisotropy99,"%f",anisotropy[y][x][j]);
			fprintf(outputAzimuth99,"%f",azimuth[y][x][j]);
			fprintf(outputAnisotropy99,"\n");
			fprintf(outputAzimuth99,"\n");
		}
		else {
			fprintf(outputAnisotropy99,"%f",anisotropy[y][x][j]);
			fprintf(outputAzimuth99,"%f",azimuth[y][x][j]);
			fprintf(outputAnisotropy99,"\t");
			fprintf(outputAzimuth99,"\t");
		}
	}

	//}

	//printf("%f",DBL_MAX);
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
	


	printf("%f\n",anisotropy[0][0][0]);

	//Freeing matrix cor
	for(i=0;i<ANGLESIZE;i++){
		free(cor[i]);
	}
	free(cor);

	//Freeing matrix cor_bi
	for(i=0;i<ANGLESIZE/2;i++){
		free(cor_bi[i]);
	}
	free(cor_bi);
	
	//Freeing matrix cmatrix
	for(i=0;i<ANGLESIZE;i++){
		free(cmatrix[i]);
	}
	free(cmatrix);

//------------------Works only when this is commented out!!---------------//
//------------------Strange as the matrices have to be freed!-------------//

	//Freeing 3D matrix anisotropy
	for(i = 0;i<YSIZE;i++) {
		for(j=0;j<XSIZE;j++) {
			free(anisotropy[i][j]);
		}
		free(anisotropy[i]);
	}
	free(anisotropy);

	//Freeing 3D matrix azimuth
	for(i = 0;i<YSIZE;i++) {
		for(j=0;j<XSIZE;j++) {
			free(azimuth[i][j]);
		}
		free(azimuth[i]);
	}
	free(azimuth);


//	free(ptrH_anisotropy);
//	free(ptrH_azimuth);
	return 0;
}
