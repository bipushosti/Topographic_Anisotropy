

#include <stdio.h>
#include <math.h>

#define XSIZE 	1200
#define YSIZE	800


#define RADIUS		100
//#define	RADWINDOW	4000
#define	RADSTEP		1
#define ANGLESIZE	72	

#define LEFT	9.892e5
#define RIGHT	1.191e6
#define UP		3.011e6
#define DOWN	2.85e6

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
	}



	//double cor[ANGLESIZE][RADIUS/RADSTEP];
	double** cor;
	cor = (double**)malloc(ANGLESIZE * sizeof(double*));
	for(i=0;i<ANGLESIZE;i++) {
		cor[i] = (double*)malloc(RADIUS/RADSTEP *sizeof(double));
	}

	
	double** cor_bi;
	cor_bi = (double**)malloc(ANGLESIZE/2 * sizeof(double*));
	for(i=0;i<ANGLESIZE/2;i++) {
		cor_bi[i] = (double*)malloc(RADIUS/RADSTEP *sizeof(double));
	}


	double*** anisotropy;
	anisotropy = (double***)malloc(YSIZE * sizeof(double**));
	for(i = 0;i<YSIZE;i++) {
		anisotropy[i] = (double**)malloc(XSIZE * sizeof(double *));
		for(j = 0; j<RADIUS;j++) {
			anisotropy[i][j] = (double*)malloc(RADIUS * sizeof(double));
		}
	}
	//double anisotropy[YSIZE][XSIZE][RADIUS];
	//float azimuth[YSIZE][XSIZE][RADIUS];


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
	
	for(i = 0;i<YSIZE;i++) {
		for(j=0;j<XSIZE;j++) {
			free(anisotropy[i][j]);
		}
		free(anisotropy[i]);
	}
	free(anisotropy);
	



	return 0;
}
