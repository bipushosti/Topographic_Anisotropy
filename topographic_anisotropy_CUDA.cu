

#include <stdio.h>
#include <math.h>
#include <float.h>

#define XSIZE 	1201
#define YSIZE	801


#define RADIUS		100
#define	RADSTEP		1
#define ANGLESIZE	72	


#define PI 3.141592653589793

int main()
{
	FILE *datTxt;
	int data[YSIZE][XSIZE];

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

	int i,j,Value;
	j = 0;
	char tempVal[5];
	memset(tempVal,'\0',sizeof(tempVal));

	while(fgets(line,1200 *5 + 2 + 1200,datTxt)!=NULL) {
	
		startPtr = line;
		//Skipping the first space
		startPtr+=1;
		//printf("%s",startPtr);

		
		for(i=0;i<XSIZE;i++) {
			endPtr = strchr(startPtr,' ');
			if(endPtr != NULL) {	
				strncpy(tempVal,startPtr,endPtr-startPtr); 
				Value = atoi(tempVal);
				data[j][i] = Value;
			}	
			endPtr = endPtr + 1;
			startPtr = endPtr;	
		}
		j++;
	}	


	float angle[ANGLESIZE];
	for(int i=0;i<ANGLESIZE;i++) {
		angle[i] = i * 5 * PI/180;
		//printf("%d	::	%f\n",i,angle[i]);
	}
	

	//Initializing 2D cmatrix
	double** cmatrix;
	cmatrix = (double**)malloc(ANGLESIZE * sizeof(double*));
	for(i=0;i<ANGLESIZE;i++) {
		cmatrix[i] = (double*)malloc(RADIUS/RADSTEP *sizeof(double));
	}

	//Initializing cor
	double** cor;
	cor = (double**)malloc(ANGLESIZE * sizeof(double*));
	for(i=0;i<ANGLESIZE;i++) {
		cor[i] = (double*)malloc(RADIUS/RADSTEP *sizeof(double));
	}

	//Initializing cor_bi
	double** cor_bi;
	cor_bi = (double**)malloc(ANGLESIZE/2 * sizeof(double*));
	for(i=0;i<ANGLESIZE/2;i++) {
		cor_bi[i] = (double*)malloc(RADIUS/RADSTEP *sizeof(double));
	}

	//Initializing 3D matrix anisotropy
	double*** anisotropy;
	anisotropy = (double***)malloc(YSIZE * sizeof(double**));
	for(i = 0;i<YSIZE;i++) {
		anisotropy[i] = (double**)malloc(XSIZE * sizeof(double *));
		for(j = 0; j<RADIUS;j++) {
			anisotropy[i][j] = (double*)malloc(RADIUS * sizeof(double));
		}
	}

	//Initializing 3D matrix anzimuth
	double*** azimuth;
	azimuth = (double***)malloc(YSIZE * sizeof(double**));
	for(i = 0;i<YSIZE;i++) {
		azimuth[i] = (double**)malloc(XSIZE * sizeof(double *));
		for(j = 0; j<RADIUS;j++) {
			azimuth[i][j] = (double*)malloc(RADIUS * sizeof(double));
		}
	}
	
	


	//Actual computation
	int xrad,yrad,x,y,k,index1,cor_bi_MinInd;
	double tempCompute,tempSum,cor_bi_ColMin;
	//for (y=0;y<YSIZE;y++) {

		//if((y>(YSIZE - RADIUS - 1))||(y<(RADIUS + 1))) continue;
	y = 0;
	for(x = 0;x<XSIZE+1;x++) {
		//printf("%d\n",y);
		//printf("Loop1\n");
		if(x==XSIZE) {
			y++;
			if(y==YSIZE){
				x = XSIZE;
				continue;
			}
			x=0;
			continue;
			
		}
		
		if((y>(YSIZE - RADIUS - 1))||(y<(RADIUS + 1))) continue;
		printf("Loop2\n");
		if((x>(XSIZE - RADIUS - 1))||(x<(RADIUS + 1))) continue;	
		printf("Loop3\n");

		for(j = 0;j<RADIUS;j+=RADSTEP) {
			for(i=0;i<ANGLESIZE;i++) {
				xrad = (int)round(cos(angle[i]) * (j+1) + x);	//<------------IT WORKS; VERIFIED
				yrad = (int)round(sin(angle[i]) * (j+1) + y);	//<------------IT WORKS; VERIFIED
	//			printf("%d) x %d	y %d\n",i+1,xrad,yrad);			
	//			printf("\t %d %d \n",(int)round(cos(angle[i]) * (j+1) + x),(int)round(sin(angle[i]) * (j+1) + y));

				cmatrix[i][j] = (double)data[yrad-1][xrad-1]; 	//<------------IT WORKS; VERIFIED
	//			printf("%d) xrad %d	yrad %d	data %f	data-1 %f\n",i+1,xrad,yrad,(double)data[yrad][xrad],(double)data[yrad-1][xrad-1]);	
	//			printf("%d) %f\n",i+1,cmatrix[i][j]);
				tempSum = 0;
				tempCompute = 0;

				for(index1 = 0;index1<=j;index1++) {					
					tempCompute = cmatrix[i][index1] - (double)data[y-1][x-1];
				//	printf("%d,%d	CM %f	DA%f\n",x,y,cmatrix[i][index1],(double)data[y-1][x-1]);
					tempCompute  = tempCompute * tempCompute;
				//	tempCompute = tempCompute / (2*j);
					tempSum = tempSum + tempCompute;
					//printf("%d,i %d,j %d) CM %f	DA %f	TS %f	",index1,i,j,cmatrix[i][index1],(double)data[y][x],tempSum);
				}
				
				cor[i][j] = tempSum/(2*(j+1));	//<------------IT WORKS; VERIFIED
				//printf("cor %f\n",tempSum/(2*(j+1)));
				printf("%d) %f\n",i+1,cor[i][j]);
				//printf("%f \n",tempSum);
			}
			return 0;
			
			cor_bi_ColMin = DBL_MAX;
			cor_bi_MinInd = 0;
			for (k=0;k<(ANGLESIZE)/2;k++) {
				cor_bi[k][j] = (cor[k][j] + cor[k+36][j])/2;
				if(cor_bi[k][j] < cor_bi_ColMin) {
					cor_bi_ColMin = cor_bi[k][j];
					cor_bi_MinInd = k;
					//printf("%f,%d\n",cor_bi_ColMin,cor_bi_MinInd);	
				}
				
			}
			int tmp;
			/*for(k=0;k<72;k++) {
				for(tmp=0;tmp<100;tmp++) {
						printf("%f ",cor[k][tmp]);
				}
				printf("\n");

			}*/
			
			
		}


		
	}
	
	//}

	//printf("%f",DBL_MAX);

	fclose(datTxt);

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
	
	return 0;
}
