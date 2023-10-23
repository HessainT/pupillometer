#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 13:08:24 2023

@author: hessain
"""
#%%
import cv2
import numpy as np

# Save the filename in an easily-accessible variable
filename = "blue_eye.jpeg"

# Define detect_ellipses function
def detect_ellipses(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale to standardize detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and help with contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use HoughCircles to detect circles in the image
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=30,
        param2=50,
        minRadius=1,
        maxRadius=600
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for i in circles[0, :]:
            # Draw the outer circle
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

        # Display the result
        cv2.imshow('Detected Ellipses', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No ellipses detected.")

# Call the detect_ellipses function with the specified filename
detect_ellipses(filename)