import numpy as np
from math import pi
from itertools import izip
from operator import itemgetter
import scipy.io 
import math
import pdb
#******************************************Variables******************************
#**Making sure the numpy arrays are float32

data = np.array([],dtype = np.float32)
xvalues = np.array([],dtype=np.float32)
yvalues = np.array([],dtype=np.float32)
zvalues = np.array([],dtype=np.float32)

#Should be int32
angle = np.array([],dtype=np.float32)
dx = np.array([],dtype=np.float32)
dy = np.array([],dtype=np.float32)
dist = np.array([],dtype=np.float32)
ind = np.array([],dtype=np.int32)

cor_bi = np.array([],dtype=np.float32)
cor = np.array([],dtype=np.float32)

semiminor = np.array([],dtype = np.float32)
tilt = np.array([],dtype = np.float32)
tmpArray = np.array([],dtype = np.float32)
aspect_ratio = np.array([],dtype = np.float32)
coords = np.array([],dtype = np.float32)

tvalx = np.array([],dtype=np.float32)
tvaly = np.array([],dtype=np.float32)

val1 = np.array([],dtype=np.float32)
val2 = np.array([],dtype=np.float32)
tmpVal2 = np.array([],dtype=np.float32)


l=0

window=0
radius=1800
radwindow=4000 
radstep=900 # measure correlation to radius advancing by radstep
angle = np.arange(0,356*pi/180,5*pi/180)[np.newaxis]#5 degree separation of spokes
angle = angle.T
angle = angle.astype(np.float32)
spoke  = np.zeros((72,1)) # initialize spoke array
#**********************************************************************************
#Loading the .mat file and getting x,y,z data
mat = scipy.io.loadmat('numbers.mat')
data = mat['dat']
data = np.float32(data)

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

cor = np.float32(cor)
cor_bi = np.float32(cor_bi)


#*************************************************************
#			Array Assignments

dist = np.zeros(shape = (len(angle),len(xvalues)))
ind = np.zeros(len(angle))
aspect_ratio = np.zeros(shape = (dataSize,radius/radstep))
semiminor = np.zeros((1,radius/radstep))
tmpArray = np.zeros(shape = (1,radius/radstep))
tilt = np.zeros((dataSize,radius/radstep))
coords = np.zeros((dataSize,3))
val1 = np.zeros(radius/radstep)
val2 = np.zeros(radius/radstep)
tmpVal2 = np.zeros(radius/radstep)

ind.dtype = np.int32
rad = np.float32()
xx = np.float32()
yy = np.float32()
cc = np.float32()
tempCC = np.float32()
#**************************************************************

kk_prime=5
kk=0


#for k in range (0,dataSize/4):
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


	if ((radius>(xx-xMin)) or 
	(radius>(yy-yMin)) or
	((radius+xx)>xMax) or 
	((radius+yy)>yMax)):		
		continue

	print "Hello\n"
	if window == 0 :
		# Step through radii
		for j in range(0,radius/radstep):
			# Advance radius length per iteration
			rad = radstep * (j+1)
			rad = np.float32(rad)

