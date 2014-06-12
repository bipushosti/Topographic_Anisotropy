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

l=0;

window=0
radius=1800
radwindow=4000 
radstep=900 # measure correlation to radius advancing by radstep
angle = np.arange(0,356*pi/180,5*pi/180)[np.newaxis]#5 degree separation of spokes.
angle = angle.T
spoke  = np.zeros(shape = (72,1)) # initialize spoke array
#**********************************************************************************


#Loading the .mat file and getting x,y,z data
mat = scipy.io.loadmat('numbers.mat')
data = mat['dat']

#Assign dimensions of study area, try to keep it small. Necessary for huge
# maps. Comment out otherwise.
#dat=fulldat
left=9.892e+5
right=1.191e+6
up=3.011e+6
down=2.85e+6

#Getting only the ones that are within the limits
data=data[data[:,0]>=left-radius,:]
data=data[data[:,0]<=right+radius,:]
data=data[data[:,1]>=down-radius,:]
data=data[data[:,1]<=up+radius,:]


if window == 1 :
	#cor has dimensions based on left, right, up, down
	#size of cor based on number of times we want to increase the scale
	#initialize raw correlogram matrix	
	cor = np.zeros((len(angle),radius/radstep-((radius-radwindow)/radstep)))
else:
	#initialize raw correlogram matrix FOR FULL AVERAGING
	cor = np.zeros((len(angle),radius/radstep)) 

cor_bi=np.zeros((len(angle)/2,radius/radstep))
print cor
#initialize mean normalized correlogram matrix
mean_norm_cor=cor



#topo X coords
xvalues = data[:,0]
#topo Y coords
yvalues = data[:,1]
#topo Z coords rounded to an integer
zvalues = data[:,2]
zvalues = zvalues.astype(int)

dataSize = len(xvalues)

kk_prime=5; 
kk=0;


for k in range(0,dataSize):
	if (k/dataSize *100 >= kk):
	    print kk,"percent done\n"
	    kk=kk+kk_prime
	
	#Current values(single values)
	xx=data[k,0]
	yy=data[k,1]
	cc=data[k,2]

	#if (radius>(xx-xvalues.min()) or 
	#radius>(yy-yvalues.min()) or
	#radius+xx>xvalues.max() or 
	#radius+yy>yvalues.max()):		
	#	continue

	l = l + 1
	# Window switch
   	#measure correlation per radius per angle, generates correlogram matrix
	#WINDOW AVERAGING

	if window == 1 :
		# Step through radii
		for j in range (0,radius/radstep-((radius-radwindow)/radstep)):
			# Advance radius length per iteration
			rad=(radius-radwindow)+radstep*(j+1)
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
				cor[i,j]=np.power(cc-zvalues[ind],2)*0.5

			if (j>0):
				#Mean of the rows gives 1 column
				temp = np.array([])
				temp = np.mean(cor[:,0:j],axis = 1)
				cor[:,j] = temp
	else:
		#For loop with j; changes the length scale
		for j in range(0,radius/radstep):
			rad = radstep * (j+1)
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
				cor[i,j]=np.power(cc-zvalues[ind],2)*0.5
			if (j>0):
				#Mean of the rows gives 1 column
				temp = np.array([])
				temp = np.mean(cor[:,0:j],axis = 1)
				cor[:,j] = temp
				
	
					
	print len(cor[:,1])
	for j in range( 0,len(cor[:,1])/2 ):
		cor_bi[j,:] = (cor[j,:]+cor[j+36,:])/2

	val1 = np.array([])
	ind = np.array([])
	val2 = np.array([])
	semiminor = np.array([])

	# How about just taking the 1-ratio of strongest correlation/orthogonal correlation
	#value?
	#Getting the minimum from the columns and storing the value in val1[]
	#	and the index in ind[]
	val1 = cor_bi.min(axis=0)

	#Looping through to get the indexes
	for j in range (0,len(cor_bi[0,:])):
		tmpVal = min(enumerate(cor_bi[:,j]),key=itemgetter(1))[0]    
		ind = np.append(ind,tmpVal)

	for i in range(0,len(ind)):
		if ind[i] <=18:
			val2 = np.append(val2,cor_bi[ind[i]+18,len(cor_bi[1,:])-1])
		else:
			val2 = np.append(val2,cor_bi[ind[i]-18,len(cor_bi[1,:])-1])

	semimajor = radius
	semiminor = radius * val1/val2




		




		
			

				

	





























