

#include <stdio.h>
#include <math.h>

#define DATASIZE 	792134
//The number of chars in x,y or z vars
#define VARSIZE		13

#define RADIUS		1800
#define	RADWINDOW	4000
#define	RADSTEP		900
#define ANGLESIZE	72	

#define LEFT	9.892e5
#define RIGHT	1.191e6
#define UP		3.011e6
#define DOWN	2.85e6

#define PI 3.141592653589793

int main()
{
	FILE *datTxt;
	//Has to be allocated since too big
	double *x,*y,*z;
	x = (double *)malloc(sizeof(double) * DATASIZE);
	y = (double *)malloc(sizeof(double) * DATASIZE);
	z = (double *)malloc(sizeof(double) * DATASIZE);

	//Max needed is VARSIZE + 6 as there are 3 spaces between 
	//vars.But 15 in case the format changes.
	char line[VARSIZE *3 + 15];
	memset(line, '\0', sizeof(line));
	char *startPtr,*endPtr;
	
	datTxt = fopen("dat.txt","r");
	if(datTxt == NULL) {
		perror("Cannot open dat.txt file");
		return (-1);
	}
	int i;
	char tempX[VARSIZE],tempY[VARSIZE],tempZ[VARSIZE];
	memset(tempX, '\0', sizeof(tempX));
	memset(tempY, '\0', sizeof(tempY));
	memset(tempZ, '\0', sizeof(tempZ));

	while(fgets(line,VARSIZE*3+10,datTxt)!=NULL) {
		//printf("%d\n",sizeof(line)/sizeof(char));
		startPtr = line;
		for(i=0;i<2;i++) {
			endPtr = strchr(startPtr,' ');
			printf("Hello\n");
			if(endPtr != NULL) {
				printf("Hello1\n");
				if(i==0) {
					strncpy(tempX,startPtr,endPtr-startPtr); 
					printf("Hello2\n");
				}
				else 
					strncpy(tempY,startPtr,endPtr-startPtr); 
					printf("Hello3\n");
			}
			
			endPtr = endPtr + 3;
			startPtr = endPtr;
			
		}
		break;
	//	endPtr = strchr(line,'\0');
	//	if(endPtr != NULL) {
	//		strncpy(tempZ,startPtr,endPtr-startPtr); 
	//	}

		
	}	

	//fclose(datTxt);
/*
	float angle[ANGLESIZE];
	for(int i=0;i<ANGLESIZE;i++) {
		angle[i] = i * 5 * PI/180;
//		printf("%f",angle[i]);
	}
	float cor[ANGLESIZE][RADIUS/RADSTEP];
	float cor_bi[ANGLESIZE/2][ANGLESIZE];
	//Setting cor to be all zeros
	memset(cor,0,sizeof(cor[0][0]) * ANGLESIZE * RADIUS/RADSTEP);
	memset(cor_bi,0,sizeof(cor_bi[0][0])* ANGLESIZE/2 * ANGLESIZE);
	printf("%f",cor[0][0]);
*/
	fclose(datTxt);
	free(x);
	free(y);
	free(z);
	return 0;
}
