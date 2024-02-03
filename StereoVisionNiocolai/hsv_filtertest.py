import cv2
import numpy as np

# Define the lower and upper bounds for the color you want to mask (in HSV color space)
color_lower = np.array([40, 50, 50])  # Lower bound for green color (adjust according to your color)
color_upper = np.array([80, 255, 255])  # Upper bound for green color (adjust according to your color)

# Initialize the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame from the video feed
    ret, frame = cap.read()

    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask based on the color range
    mask = cv2.inRange(hsv, color_lower, color_upper)

    # Apply morphological operations to remove noise and refine the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Bitwise-AND the mask with the original frame to extract the colored region
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the original frame and the masked result
    cv2.imshow("Original", frame)
    cv2.imshow("Masked Result", result)

    # If 'q' is pressed, break from the loop
    if cv2.waitKey(1) == ord("q"):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
