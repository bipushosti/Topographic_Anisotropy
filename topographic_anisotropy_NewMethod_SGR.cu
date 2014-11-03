#include <stdio.h>
#include <math.h>
#include <float.h>
#include <limits.h>

#define XSIZE 	1201
#define YSIZE	801



#define RADIUS		100
#define	RADSTEP		1
#define ANGLESIZE	36	


#define PI 3.141592653589793

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
	
	//Fine
	float angle[ANGLESIZE];
	for(int i=0;i<ANGLESIZE;i++) {
		angle[i] = i * 5 * PI/180;
	}
	
	//Initializing 3D matrix anisotropy
	float*** anisotropy;
	anisotropy = (float***)malloc(YSIZE * sizeof(float**));
	for(i = 0;i<YSIZE;i++) {
		anisotropy[i] = (float**)malloc(XSIZE * sizeof(float *));
		for(j = 0; j<XSIZE;j++) {
			anisotropy[i][j] = (float*)malloc(RADIUS * sizeof(float));
		}
	}

	//Initializing 3D matrix anzimuth
	float*** azimuth;
	azimuth = (float***)malloc(YSIZE * sizeof(float**));
	for(i = 0;i<YSIZE;i++) {
		azimuth[i] = (float**)malloc(XSIZE * sizeof(float *));
		for(j = 0; j<XSIZE;j++) {
			azimuth[i][j] = (float*)malloc(RADIUS * sizeof(float));
		}
	}

	//Actual computation
	int xrad,yrad,x,y,xradOrtho1,yradOrtho1,xradOneEighty,yradOneEighty,valueOneEighty;
	int valueOrtho1,valueOrtho2,xradOrtho2,yradOrtho2;
	float variance[100];
	float orientation[100];
	float ortho[100];
	float value,sum_value,avg_value;
	float sum_valueOrtho,avg_valueOrtho;
	sum_value = 0;
	avg_value = 0;
	sum_valueOrtho = 0;
	avg_valueOrtho = 0;

	

	//y = 0;
	for(y=0;y<YSIZE;y++) {
		for(x = 0;x<XSIZE;x++) {
		/*for(x = 0;x<XSIZE+1;x++) {
			if(x==XSIZE) {
				y++;	
				if(y==YSIZE){
					x = XSIZE;
					continue;
				}
				x=0;
				continue;
			
			}
			*/
			if((y>(YSIZE - RADIUS - 1))||(y<(RADIUS))) continue;
			if((x>(XSIZE - RADIUS - 1))||(x<(RADIUS))) continue;	

			for(i=0;i<100;i++){
				variance[i] = FLT_MAX;
				ortho[i] = FLT_MAX;
			}
			
			
			//Flipped
			for(i=0;i<ANGLESIZE;i++) {
				//Initializing to 0 so that the sum is zero everytime it starts
				sum_value = 0;
				sum_valueOrtho = 0;
				for(j = 0;j<RADIUS;j+=RADSTEP) {
		
					//Computation for angle of interest
					xrad = (int)round(cos(angle[i]) * (j+1) + x);	
					yrad = (int)round(sin(angle[i]) * (j+1) + y);	

					value = data[y][x] - data[yrad][xrad];
					value = value * value;
					
					//One eighty angle computation
					xradOneEighty = (int)round(cos(angle[i]+PI) * (j+1) + x);	
					yradOneEighty = (int)round(sin(angle[i]+PI) * (j+1) + y);	
					
					valueOneEighty = data[y][x] - data[yradOneEighty][xradOneEighty];
					valueOneEighty = valueOneEighty * valueOneEighty;

					sum_value = sum_value + value + valueOneEighty;
					avg_value = sum_value/(2*(j+1)); //the average variance from scale 1 to scale j

					//Computation for values on angle orthogonal to angle of interest
					xradOrtho1 = (int)round(cos(angle[i]+PI/2) * (j+1) + x);	
					yradOrtho1 = (int)round(sin(angle[i]+PI/2) * (j+1) + y);	
					
					valueOrtho1 = data[y][x] - data[yradOrtho1][xradOrtho1];
					valueOrtho1 = valueOrtho1 * valueOrtho1;

					//One eighty ortho angle computation
					xradOrtho2 = (int)round(cos(angle[i]+PI*3/2) * (j+1) + x);	
					yradOrtho2 = (int)round(sin(angle[i]+PI*3/2) * (j+1) + y);	

					valueOrtho2 = data[y][x] - data[yradOrtho2][xradOrtho2];
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
					
					//printf("1(%d,%d)	%f	%f\n",(j+1),(i+1),variance[j],avg_value);
					//Determine if the variance is minimum compared to  others at scale j, if so record it and its angle i. If not, pass it
					if(avg_value < variance[j]) {
						//	printf("2(%d)	%f	%f\n",j,variance[j],avg_value);
							variance[j] = avg_value;
							orientation[j] = angle[i];
							ortho[j] = avg_valueOrtho;		
					}	
				}
			}
			
			//variance, ortho, and orientation arrays should represent the orientation of the minimum variance for all scales. Take ratio of ortho to variance to get anisotropy and convert orientation to degrees.
			for(j=0;j<RADIUS;j+=RADSTEP){
				anisotropy[y][x][j] = ortho[j]/variance[j];
				azimuth[y][x][j] = orientation[j] * 180/PI ;
				
				//printf("%f	%f\n",variance[j],anisotropy[y][x][j]);	
			}

			
//			Writing to files
			

			if (x == (XSIZE - RADIUS - 1)) {
				fprintf(outputAnisotropy00,"%f",anisotropy[y][x][0]);
				fprintf(outputAzimuth00,"%f",azimuth[y][x][0]);
				fprintf(outputAnisotropy00,"\n");
				fprintf(outputAzimuth00,"\n");

				fprintf(outputAnisotropy09,"%f",anisotropy[y][x][9]);
				fprintf(outputAzimuth09,"%f",azimuth[y][x][9]);
				fprintf(outputAnisotropy09,"\n");
				fprintf(outputAzimuth09,"\n");

				fprintf(outputAnisotropy49,"%f",anisotropy[y][x][49]);
				fprintf(outputAzimuth49,"%f",azimuth[y][x][49]);
				fprintf(outputAnisotropy49,"\n");
				fprintf(outputAzimuth49,"\n");

				fprintf(outputAnisotropy99,"%f",anisotropy[y][x][99]);
				fprintf(outputAzimuth99,"%f",azimuth[y][x][99]);
				fprintf(outputAnisotropy99,"\n");
				fprintf(outputAzimuth99,"\n");
			}
			else {
				fprintf(outputAnisotropy00,"%f",anisotropy[y][x][0]);
				fprintf(outputAzimuth00,"%f",azimuth[y][x][0]);
				fprintf(outputAnisotropy00,"\t");
				fprintf(outputAzimuth00,"\t");
				
				fprintf(outputAnisotropy09,"%f",anisotropy[y][x][9]);
				fprintf(outputAzimuth09,"%f",azimuth[y][x][9]);
				fprintf(outputAnisotropy09,"\t");
				fprintf(outputAzimuth09,"\t");

				fprintf(outputAnisotropy49,"%f",anisotropy[y][x][49]);
				fprintf(outputAzimuth49,"%f",azimuth[y][x][49]);	
				fprintf(outputAnisotropy49,"\t");
				fprintf(outputAzimuth49,"\t");

				fprintf(outputAnisotropy99,"%f",anisotropy[y][x][99]);
				fprintf(outputAzimuth99,"%f",azimuth[y][x][99]);
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
