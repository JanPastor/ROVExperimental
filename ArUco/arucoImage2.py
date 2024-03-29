import cv2
import imutils
import sys


# Note this version requires no command line parsing

# Specify the file path of the image and the type of ArUco dictionary
image_path = "ArUco/aruco_guess.jpg"
aruco_dictionary_type = "DICT_5X5_100"

# Define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# Load the input image from disk and resize it
print("[INFO] loading image...")
image = cv2.imread(image_path)
image = imutils.resize(image, width=600)

# Verify that the supplied ArUCo tag exists and is supported by OpenCV
if ARUCO_DICT.get(aruco_dictionary_type, None) is None:
    print("[INFO] ArUCo tag of '{}' is not supported".format(aruco_dictionary_type))
    sys.exit(0)

# Load the ArUCo dictionary, grab the ArUCo parameters, and detect the markers
print("[INFO] detecting '{}' tags...".format(aruco_dictionary_type))
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_dictionary_type])
arucoParams = cv2.aruco.DetectorParameters()
(corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

# Verify *at least* one ArUco marker was detected
if len(corners) > 0:
    # Flatten the ArUco IDs list
    ids = ids.flatten()

    # Loop over the detected ArUCo corners
    for (markerCorner, markerID) in zip(corners, ids):
        # Extract the marker corners (which are always returned in top-left, top-right, bottom-right, and bottom-left order)
        corners = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners

        # Convert each of the (x, y)-coordinate pairs to integers
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))

        # Draw the bounding box of the ArUCo detection
        cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
        cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
        cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
        cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

        # Compute and draw the center (x, y)-coordinates of the ArUco marker
        cX = int((topLeft[0] + bottomRight[0]) / 2.0) # Compute X centroid
        cY = int((topLeft[1] + bottomRight[1]) / 2.0) # Compute Y centroid
        cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1) # Draw a red circle at the centroid location

        # Draw the ArUco marker ID on the image
        cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print("[INFO] ArUco marker ID: {}".format(markerID))

    # Show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
