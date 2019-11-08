#!/bin/bash
for i in *.svg
do
	filename=$(basename -- "$i")
	extension="${filename##*.}"
	filename="${filename%.*}"
	inkscape -z -e $filename".png" $filename".svg" 
done

