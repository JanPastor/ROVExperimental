import cv2
import numpy as np
import time

def add_HSV_filter(frame, camera):
    # Define the lower and upper bounds for the color you want (in HSV color space)
    color_lower = np.array([40, 40, 40])  # Lower limit for green bowl
    color_upper = np.array([70, 255, 255])  # Upper limit for green bowl

    # Blur the frame
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    # Converting the frame from BGR to HSV color space
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, color_lower, color_upper)

    # Apply morphological operations to remove noise and refine the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # # Bitwise-AND the mask with the original frame to extract the colored region
    # result = cv2.bitwise_and(frame, frame, mask=mask)

    return mask
    # return result
