#!/bin/bash

trap

# Delay in seconds between images
DELAY=20

# Loop through all image files in the directory
while true; do
	for image in vis/3d_gl/*; do
	    if [ -f "$image" ]; then
		echo $image
		eog "$image" --single-window --fullscreen &  # Display the image in fullscreen and fork to background
		sleep "$DELAY"     # Wait for the specified delay
	    fi
	done
done
