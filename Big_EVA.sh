#!/bin/bash


RADIUS=100
DELIMITER='space'

#------------------------------------
FILE="./Annie/map_"
EXT=".txt"
#------------------------------------
make clean
make 

run_program()
{	
	NUM=0
		
	FULL_FILE=$FILE$NUM$EXT
	echo $FULL_FILE

	while [ -f $FULL_FILE ]
	do
		#echo "File exists"
		echo $FULL_FILE
		
		./MultiGPU $FULL_FILE $DELIMITER $RADIUS
		((NUM++))
		FULL_FILE=$FILE$NUM$EXT				
	done

}
run_program

echo "End of program!"

exit $?


