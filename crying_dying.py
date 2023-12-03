
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:18:53 2023

@author: oumousamake
"""


#%%
import cv2
import numpy as np
import math

def detect_eye(img):
    # Resize the image
    scaling_factor = 0.7
    img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # Display the original image
    cv2.imshow('Input', img)

    # Convert the image to grayscale without color inversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh_gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Iterate through contours
    for contour in contours:
        area = cv2.contourArea(contour)
        rect = cv2.boundingRect(contour)
        x, y, width, height = rect
        radius = 0.25 * (width + height)
    
        # Experiment with different criteria for filtering contours
        area_condition = (50 <= area <= 500)
        aspect_ratio_condition = (0.8 <= width/height <= 1.2)
    
        if area_condition and aspect_ratio_condition:
            cv2.circle(img, (int(x + radius), int(y + radius)), int(1.3*radius), (0, 180, 0), -1)

    return img  # Return the processed image

# Open a connection to the default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Main loop for capturing and processing video
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Perform eye detection on the current frame
    frame_with_eyes = detect_eye(frame)

    # Display the processed frame with eye detection
    cv2.imshow('Eye Detection', frame_with_eyes)

    # Check for the 'q' key to exit the loop
    code = cv2.waitKey(10)
    if code == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


