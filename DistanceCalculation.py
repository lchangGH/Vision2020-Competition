# import the necessary packages
from imutils import paths
import numpy as np
import imutils
import cv2

def biggestContourI(contours):
    maxVal = 0
    maxI = -1
    for i in range(0, len(contours)):
        if len(contours[i]) > maxVal:
            cs = contours[i]
            maxVal = len(contours[i])
            maxI = i
    return maxI

def find_marker(image):
	# convert the image to hsvscale, blur it, and detect edges
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    contours0, hierarchy = cv2.findContours(flt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Only draw the biggest one
    bc = biggestContourI(contours0)
    print("bc " + str(bc))
    #print("contours " + str(contours0))
    if bc != -1 :
        cv2.drawContours(image,contours0, bc, (220,0,255), 3)

    lower_yellow = np.array([20,60,50])
    upper_yellow = np.array([90,255,255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    res = cv2.bitwise_and(image,image, mask= mask)
    
    cv2.imshow('my webcam', image)
    cv2.imshow('hsv', hsv)
    #cv2.imshow('flt', flt)
    cv2.imshow('res',res)


# distance from camera to object using Python and OpenCVPython
def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 24.0
 
# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 11.0
 
# load the furst image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length
image = cv2.imread("C:\VSCodeMain\Vision2019\Vision2020\Vision2020-Competition\PowerCell25Scale\power+02f.jpg")
marker = find_marker(image)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
print ("I worked!")

# loop over the images
while (True):
	# load the image, find the marker in the image, then compute the
	# distance to the marker from the camera
	image = cv2.imread("C:\VSCodeMain\Vision2019\Vision2020\Vision2020-Competition\PowerCellImages\power+02f.jpg")
	marker = find_marker(image)
	inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])

	# draw a bounding box around the image and display it
	box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
	box = np.int0(box)
	cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
	cv2.putText(image, "%.2fft" % (inches / 12),
		(image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
		2.0, (0, 255, 0), 3)
	cv2.imshow("image", image)
	cv2.waitKey(0)

