#!/usr/bin/python

import numpy as np
import gnumpy as gp
from math import pi
from itertools import izip
from operator import itemgetter
import scipy.io 
import math
import pdb
from datetime import datetime



#****************************************Function Definitions*********************
#Function to get dx and dy components
#Type 0=dx 1=dy
def get_dx_dy(Var,VarArray,rad,index,Type):
	VarArray = gp.garray(VarArray)

	if(Type == 0):
		VarRad = np.cos(angle[index])*rad
	else:
		VarRad = np.sin(angle[index])*rad

	VarRad = gp.garray(VarRad)
	dVar = gp.garray([])	
	dVar = (Var + VarRad) - VarArray
	return dVar.as_numpy_array()


#Function to get the distance using dx and dy
def get_dist(dvar1,dvar2):
	dvar1 = gp.garray(dvar1)
	dvar2 = gp.garray(dvar2)
	dist = (dvar1**2 + dvar2**2)**0.5
	return dist.as_numpy_array()


#Function to get the cor values
def get_cor(CorArray,ValArray,MinIndex,Val,IndexI,IndexJ):
	CorArray = gp.garray(CorArray)
	ValArray  = gp.garray(ValArray)
	tmpVal = pow(Val - ValArray[MinIndex],2)*0.5
	CorArray[IndexI,IndexJ] = tmpVal
	return CorArray.as_numpy_array()

	
#Function to get the mean of the rows
def get_row_mean(ValArray,IndexJ):
	ValArray = gp.garray(ValArray)
	ValArray[:,j] = gp.mean(ValArray[:,0:IndexJ+1],axis = 1)
	return ValArray.as_numpy_array()

#Function to get the array cor_bi
def get_cor_bi(CorBiArray,CorArray,Index):
	CorArray = gp.garray(CorArray)
	CorBiArray = gp.garray(CorBiArray)
	CorBiArray[Index,:] = (CorArray[Index,:]+CorArray[Index+36,:])/2
	return CorBiArray.as_numpy_array()
	

#Function to get the semiminor
def get_semiminor(ArrayVal1,ArrayVal2,Radius):
	SemiminorArray = gp.garray([])
	ArrayVal1 = gp.garray(ArrayVal1)
	ArrayVal2 = gp.garray(ArrayVal2)
	SemiminorArray = radius * ArrayVal1/ArrayVal2
	return SemiminorArray.as_numpy_array()

	
#Function to get the aspect ratio
def get_aspect_ratio(SemiminorArray,Semimajor):
	AspectRatioArray = gp.garray([])
	SemiminorArray = gp.garray(SemiminorArray)
	#SemimajorArray = gp.garray(SemimajorArray)
	AspectRatioArray = 1 - ((SemiminorArray**2)/(Semimajor*Semimajor))**0.5
	return AspectRatioArray.as_numpy_array()


	
	
#****************************Variables & Array Declarations**********************

#**Making sure the numpy arrays are float32
xvalues = np.array([],dtype='f')
yvalues = np.array([],dtype='f')
zvalues = np.array([],dtype='f')

angle = np.array([],dtype='f')
dx = np.array([],dtype='f')
dy = np.array([],dtype='f')
dist = np.array([],dtype='f')

cor_bi = np.array([],dtype='f')
cor = np.array([],dtype='f')

semiminor = np.array([],dtype = 'f')
tilt = np.array([],dtype = 'f')
aspect_ratio = np.array([],dtype = 'f')
coords = np.array([],dtype = 'f')





l=0

window=0
radius=1800
radwindow=4000 
radstep=900 # measure correlation to radius advancing by radstep

angle = np.arange(0,356*pi/180,5*pi/180)[np.newaxis]#5 degree separation of spokes.
#SinAngle
#CosAngle
angle = angle.T
#angle = gp.garray(angle)
spoke  = gp.zeros((72,1)) # initialize spoke array


#**********************************************************************************
#Loading the .mat file and getting x,y,z data
mat = scipy.io.loadmat('numbers.mat')
data = mat['dat']

#Assign dimensions of study area, try to keep it small. Necessary for huge
# maps. Comment out otherwise.
#dat=fulldat
left=9.892e5
right=1.191e6
up=3.011e6
down=2.85e6

#Getting only the ones that are within the limits
data=data[data[:,0]>=left-radius,:]
data=data[data[:,0]<=right+radius,:]
data=data[data[:,1]>=down-radius,:]
data=data[data[:,1]<=up+radius,:]

#topo X coords
xvalues = data[:,0]
#topo Y coords
yvalues = data[:,1]
#topo Z coords rounded to an integer
zvalues = data[:,2]
#zvalues = zvalues.astype(int)

dataSize = len(xvalues)


xMin = np.amin(xvalues)
xMax = np.amax(xvalues)
yMin = np.amin(yvalues)
yMax = np.amax(yvalues)


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



#*************************************************************
#			Array Assignments
aspect_ratio = np.zeros(shape = (dataSize,radius/radstep))
semiminor = np.zeros((1,radius/radstep))
tilt = np.zeros((dataSize,radius/radstep))
coords = np.zeros((dataSize,3))
#**************************************************************

kk_prime=5
kk=0



#for k in range(0,dataSize):
for k in range(555,755):
	if (k/dataSize *100 >= kk):
	    print kk,"percent done\n"
	    kk=kk+kk_prime
	
	print k
	#Current values(single values)
	xx=data[k,0]
	yy=data[k,1]
	cc=data[k,2]
	
	#print "radius is:"
	#print radius
	#print "\n"
	#print xx - xMin
	#print "\n"
	#print yy - yMin
	#print "\n"
	#print xMax - xx
	#print "\n"
	#print yMax - yy
	#print "\n"


	if ((radius>(xx-xMin)) or 
	(radius>(yy-yMin)) or
	((radius+xx)>xMax) or 
	((radius+yy)>yMax)):		
		continue

	xx = gp.garray(xx)
	yy = gp.garray(yy)
	cc = gp.garray(cc)
	
	print "Hello\n"
	if window == 0 :
		# Step through radii
		for j in range(0,radius/radstep):
			
			# Advance radius length per iteration
			rad = radstep * (j+1)

			for i in range(0,len(angle)):				
				dx = get_dx_dy(xx,xvalues,rad,i,0)
				dy = get_dx_dy(yy,yvalues,rad,i,1)
				dist = get_dist(dx,dy)
				#val = distance to the closest point ,ind = index of that point
				# Find point closest to the radius value j for each angle
				val = min(dist)
				ind = dist.argmin(axis=0)
				cor = get_cor(cor,zvalues,ind,cc,i,j)

			if (j>0):
				cor = get_row_mean(cor,j)

	else:
		for j in range(0,radius/radstep-((radius-radwindow)/radstep)):
			rad = (radius-radwindow)+radstep*(j+1)

			for i in range(0,len(angle)):
				dx = get_dx_dy(xx,xvalues,rad,i,0)
				dy = get_dx_dy(yy,yvalues,rad,i,1)
				dist = get_dist(dx,dy)
				val = min(dist)
				ind = dist.argmin(axis=0)
				cor = get_cor(cor,zvalues,ind,cc,i,j)

			if (j>0):
				cor = get_row_mean(cor,j)
	
	for j in range (0,len(cor[:,1])/2):
		cor_bi = get_cor_bi(cor_bi,cor,j)

	ind = np.array([])
	val2 = np.array([])
	val1 = np.array([])	

	# How about just taking the 1-ratio of strongest correlation/orthogonal correlation
	#value?
	#Getting the minimum from the columns and storing the value in val1[]
	#	and the index in ind[]
	val1 = cor_bi.min(axis = 0)
	ind = cor_bi.argmin(axis = 0)

	
	for i in range(0,len(ind)):
		if ind[i] <=18:
			val2 = np.append(val2,cor_bi[ind[i]+18,len(cor_bi[1,:])-1])
		else:
			val2 = np.append(val2,cor_bi[ind[i]-18,len(cor_bi[1,:])-1])
	
	semimajor = radius
	semiminor = get_semiminor(val1,val2,radius)
	aspect_ratio[l,:] = get_aspect_ratio(semiminor,semimajor)
	l = l + 1

