import sys
import cv2
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt
import cameraParams2

# Functions to be imported
import HSV_filter2 as hsv
import shape_recognition as shape
import triangulation as tri


def undistort_frames(frame_L, frame_R):
    # Access the camera parameters
    camera_matrix = cameraParams2.camera_matrix
    distortion_coeffs = cameraParams2.dist_coeffs
    
    # Undistort the left and right frames
    undist_left_frame = cv2.undistort(frame_L, camera_matrix, distortion_coeffs)
    undist_right_frame = cv2.undistort(frame_R, camera_matrix, distortion_coeffs)

    return undist_left_frame, undist_right_frame

# Initialize and open both cameras
cap_right = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap_left = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Define frame rate
frame_rate = 120
# Define camera parameters
B = 14 # Distance between cameras in (cm)
f = 3 # Camera lens's focal length (mm) (Assuming focal length of lens)
alpha = 170 # Camera field of view in the horizontal plane (degrees)

# Initial values
count = -1

while True:
    count += 1 
    ret_right, frame_right = cap_right.read()
    ret_left, frame_left = cap_left.read()

    # Undistortion Step
    undist_left_frame, undist_right_frame = undistort_frames(frame_left, frame_right)

# If cannot catch any frame, break
    if ret_right == False or ret_left == False:
        break
    else:
        # Apply HSV-Filter:
        mask_right = hsv.add_HSV_filter(undist_right_frame, 0)
        mask_left = hsv.add_HSV_filter(undist_left_frame, 1)
        # Result-frames after applying HSV filter mask
        res_right = cv2.bitwise_and(undist_right_frame, undist_right_frame, mask = mask_right)
        res_left = cv2.bitwise_and(undist_left_frame, undist_left_frame, mask= mask_left)

        # Applying shape recognition:
        circles_right = shape.find_circles(undist_right_frame, mask_right)
        circles_left = shape.find_circles(undist_left_frame, mask_left)
        # Also apply shape recognition for rectangles?

    # Calculating Ball Depth

    # If no ball can be caught in one camera show text "Tracking Lost"

    if np.all(circles_right) == None or np.all(circles_left) == None:
        cv2.putText(undist_right_frame, "Tracking Lost", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(undist_left_frame, "Tracking Lost", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    else:
        # Function to calculate depth of object: Outputs vector of all depths in case...
        # All formulas used to find to depth is in video presenation
        depth = tri.find_depth(circles_right, circles_left, undist_right_frame, undist_left_frame,B, f, alpha)
        cv2.putText(undist_right_frame, "Tracking", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
        cv2.putText(undist_left_frame, "Tracking", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
        cv2.putText(undist_right_frame, "Distance: " + str(round(depth,3)), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
        cv2.putText(undist_left_frame, "Distance: " + str(round(depth, 3)), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
        # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor is obtained by 

        print("Depth:", depth)

    # Show the frames
    cv2.imshow("frame right", undist_right_frame)
    cv2.imshow("frame left", undist_left_frame)
    cv2.imshow("mask right", res_right)
    cv2.imshow("mask left", res_left)

    # Hit "q" to close th window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and destroy all windows before termination

cap_right.release()
cap_left.release()
cv2.destroyAllWindows()
   
    