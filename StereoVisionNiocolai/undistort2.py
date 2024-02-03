import cv2
import numpy as np
import cameraParams2

def undistort_frames(left_frame, right_frame):
    # Access the camera parameters
    camera_matrix = cameraParams2.camera_matrix
    distortion_coeffs = cameraParams2.dist_coeffs

    # Undistort the left and right frames
    undistorted_left_frame = cv2.undistort(left_frame, camera_matrix, distortion_coeffs)
    undistorted_right_frame = cv2.undistort(right_frame, camera_matrix, distortion_coeffs)

    return undistorted_left_frame, undistorted_right_frame
