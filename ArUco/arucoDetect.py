import cv2
import numpy as np

# Initialize the camera capture object
cap = cv2.VideoCapture(0)

# Load the ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

while True:
    # Read the current frame from the camera feed
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the grayscale frame
    arucoParams = cv2.aruco.DetectorParameters()
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=arucoParams)
    if corners:
        # Draw bounding boxes around the detected markers
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))

        # Draw polygon around the marker (use the marker_masked_frame)
        int_corners = np.int32(corners)
        cv2.polylines(frame, int_corners, True, (255, 255, 255), 2)

        # Calculate bounding rectangle
        x, y, w, h = cv2.boundingRect(int_corners)

        cv2.putText(frame, "Width {} cm".format(round(x, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2) 
        cv2.putText(frame, "Height {} cm".format(round(y, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)  
        # Display the frame with bounding boxes
        cv2.imshow('ArUco Marker Detection', frame)

        # Check for the 'q' key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera capture object and close all windows
cap.release()
cv2.destroyAllWindows()
