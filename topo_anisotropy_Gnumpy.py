
import numpy as np
import gnumpy as gp
from math import pi
from itertools import izip
from operator import itemgetter
import scipy.io 
import math
import pdb
from datetime import datetime
import gnumpy as gpu




#******************************************Variables******************************

#All the x,y and z values 
#Called x,y,c in the MATLAB script
xvalues = np.array([],dtype='f')
yvalues = np.array([],dtype='f')
zvalues = np.array([],dtype='f')
angle = np.array([],dtype='f')

dx = np.array([],dtype='f')
dy = np.array([],dtype='f')
#dist = np.array([],dtype='f')
#aspect_ratio = np.array([],dtype='f')
#tilt = np.array([],dtype='f')
#coords = np.array([],dtype='f')

dx = gp.garray(dx)
dy = gp.garray(dy)
#dist = gp.garray(dist)
#aspect_ratio = gp.garray(aspect_ratio)
#tilt = gp.garray(tilt)
#coords = gp.garray(coords)

l=0;

window=0
radius=1800
radwindow=4000 
radstep=900 # measure correlation to radius advancing by radstep

angle = np.arange(0,356*pi/180,5*pi/180)[np.newaxis]#5 degree separation of spokes.
#SinAngle
#CosAngle
angle = angle.T
angle = gp.garray(angle)
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

xvalues = gp.garray(xvalues)
yvalues = gp.garray(yvalues)
zvalues = gp.garray(zvalues)

if window == 1 :
	#cor has dimensions based on left, right, up, down
	#size of cor based on number of times we want to increase the scale
	#initialize raw correlogram matrix	
	cor = gp.zeros((len(angle),radius/radstep-((radius-radwindow)/radstep)))
else:
	#initialize raw correlogram matrix FOR FULL AVERAGING
	cor = gp.zeros((len(angle),radius/radstep)) 

cor_bi=gp.zeros((len(angle)/2,radius/radstep))

#initialize mean normalized correlogram matrix
mean_norm_cor=cor



#*************************************************************
#			Array Declarations

tmpArray = gp.zeros((1,radius/radstep))
dist = np.array([],dtype='f')
dist = gp.garray(dist)
semiminor = gp.zeros((1,radius/radstep))
tilt = gp.zeros((dataSize,radius/radstep))
aspect_ratio = gp.zeros((dataSize,radius/radstep))
coords = gp.zeros((dataSize,3))
#**************************************************************

kk_prime=5; 
kk=0;



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
	xx = gp.garray(xx)
	yy = gp.garray(yy)
	cc = gp.garray(cc)

	if ((radius>(xx-xMin)) or 
	(radius>(yy-yMin)) or
	((radius+xx)>xMax) or 
	((radius+yy)>yMax)):		
		continue
	print "Hello\n";
	if window == 0 :
		# Step through radii
		for j in range (0,radius/radstep-((radius-radwindow)/radstep)):
			# Advance radius length per iteration
			rad=(radius-radwindow)+radstep*(j+1)
			for i in range(0,len(angle)):
				xrad=gp.cos(angle[i])*rad # Find adjacent length (x component)
				yrad=gp.sin(angle[i])*rad # Find opposite length (y component)
				#dx = (xx + xrad) - xvalues
				#dy = (yy+yrad)-yvalues
				dx = xx+xrad
				dx = (xx+xrad[0]) - xvalues
				dy = yvalues


				#val = distance to the closest point 
				#ind = index of that point
				# Find point closest to the radius value j for each angle
	#			val = min(enumerate(dist),key=itemgetter(1))[1]
	#			ind = min(enumerate(dist),key=itemgetter(1))[0]
				#Measure correlation and stick it in an angle x radius matrix
	#			cor[i,j]=gp.power(cc-zvalues[ind],2)*0.5





