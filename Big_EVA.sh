#!/bin/bash


RADIUS=100
DELIMITER='space'

#------------------------------------
FILE="./Maine/map_"
EXT=".txt"
#------------------------------------

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


