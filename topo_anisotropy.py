#! /usr/bin/python

import numpy as np
from math import pi
from itertools import izip
from operator import itemgetter
import scipy.io 
import math



#******************************************Variables******************************

#All the x,y and z values 
#Called x,y,c in the MATLAB script
xvalues = np.array([])
yvalues = np.array([])
zvalues = np.array([])

dx = np.array([])
dy = np.array([])
dist = np.array([])

#Arrays containing the position data of the 
#study area
xArea=[]
yArea=[]
zArea=[]


tilt=[]
aspect_ratio=[]
coords=[]


kk_prime=5; 
kk=0;
l=0;
#**********************************************************************************


#Loading the .mat file and getting x,y,z data
mat = scipy.io.loadmat('numbers.mat')
data = mat['dat']
xvalues = data[:,0]
yvalues = data[:,1]
zvalues = data[:,2]
zvalues = zvalues.astype(int)

dataSize = len(xvalues)


window=0
radius=1800
radwindow=4000 
radstep=900 # measure correlation to radius advancing by radstep
angle = np.arange(0,355*pi/180,5*pi/180)[np.newaxis]#5 degree separation of spokes.
angle = angle.T
spoke  = np.zeros(shape = (71,1)) # initialize spoke array




#Assign dimensions of study area, try to keep it small. Necessary for huge
# maps. Comment out otherwise.
#dat=fulldat
left=9.892e+5
right=1.191e+6
up=3.011e+6
down=2.85e+6

for i in range(0,dataSize):
	if(xvalues[i]>=(left-radius) and 
	xvalues[i]<=(left+radius) and
	yvalues[i]>=(down-right) and
	yvalues[i]<=(up+radius)):
		xArea.append(xvalues[i])
		yArea.append(yvalues[i])
		zArea.append(zvalues[i])

if window == 1 :
    #cor has dimensions based on left, right, up, down
    #size of cor based on number of times we want to increase the scale
	#initialize raw correlogram matrix	
    cor = np.zeros((len(angle),radius/radstep-((radius-radwindow)/radstep)))
else:
	#initialize raw correlogram matrix FOR FULL AVERAGING
    cor = np.zeros((len(angle),radius/radstep)) 

cor_bi=np.zeros((len(angle)/2,radius/radstep))
#initialize mean normalized correlogram matrix
mean_norm_cor=cor

for k in range(0,dataSize):
    if (k/dataSize *100 >= kk):
        print kk,"percent done\n"
        kk=kk+kk_prime

    #Current values(single values)
	xx=data[k,0]
	yy=data[k,1]
	cc=data[k,2]

	if (radius>(xx-xvalues.min()) or 
	radius>(yy-yvalues.min()) or
	radius+xx>xvalues.max() or 
	radius+yy>yvalues.max()):		
		continue

	l = l + 1
	# Window switch
   	#measure correlation per radius per angle, generates correlogram matrix
    #WINDOW AVERAGING

	if window == 1 :
		# Step through radii
		for j in range (0,radius/radstep-((radius-radwindow)/radstep)):
			# Advance radius length per iteration
			rad=(radius-radwindow)+radstep*j
			# Step through angles, 5 degree intervals
			for i in range(0,len(angle)):
				xrad=math.cos(angle[i])*rad # Find adjacent length (x component)
				yrad=math.sin(angle[i])*rad # Find opposite length (y component)
				dx = (xx + xrad) - xvalues
				dy = (yy+yrad)-yvalues
				dist = np.power(np.power(dx,2) + np.power(dy,2),0.5)
				#val = distance to the closest point 
				#ind = index of that point
				# Find point closest to the radius value j for each angle i			
				val = min(enumerate(dist),key=itemgetter(1))[1]
				ind = min(enumerate(dist),key=itemgetter(1))[0]
				#Measure correlation and stick it in an angle x radius matrix
				cor[i,j]=np.power(cc-c[ind],2)*0.5

		if (j>0):
			#Mean of the rows gives 1 column
			temp = np.array([])
			temp = (np.mean(cor[:,0:(j+1)],axis = 1))
			temp.shape = (len(temp),1)
			cor[:,j] = temp

	for j in range( 0,len(cor[:,1])/2 ):
		cor_bi[j,:] = (cor[j,:]+cor[j+36,:])/2

	val1 = np.array([])
	ind = np.array([])
	val2 = np.array([])
	semiminor = np.array([])
	
	#val1 = 	min(enumerate(cor_bi),key=itemgetter(1))[1]	
	val1 = cor_bi.min(axis=0)
	for j in range (0,len(cor_bi)):
		tmpVal = min(enumerate(cor_bi[:,j]),key=itemgetter(1))[1]    
		ind = np.append(ind,tmpVal)
	
			




			
				

				

	





























