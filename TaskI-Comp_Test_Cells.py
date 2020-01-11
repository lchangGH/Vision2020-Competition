# This is a pseudo code file for Merge Robotics, 2020, Infinite Recharge
# This is task I - > OpenCV "Contours Continued."  Not sure if it is clear by now, 
# but OpenCV can do a lot of things, we need to understand what it offers to complete 
# our vision code.  For a given single contour, (meaning it was imaged and masked and 
# converted to a coordinate array), you need to be able to use a number of OpenCV functions.
# Please experiment with the following, easiest is to simply draw them back to a blank image
# or on top of original.  Some very important OpenCV produced actions van only br
# printed to the console or used in code to filter contours.

# Continuing from last week - contour perimeter, contour approximation, bounding rectangles, 
# minimum enclosing circle, fitting elipse, fitting line, aspect ratio
# extent, solidity, equivalent diameter, orientation, points, min/max
# mean color, extreme points

# useful links
# https://docs.opencv.org/3.4.7/dd/d49/tutorial_py_contour_features.html
# https://docs.opencv.org/3.4.7/d1/d32/tutorial_py_contour_properties.html


# imports
import numpy as np
import cv2 
from pathlib import Path
import sys

print('Using python version {0}'.format(sys.version))
print('Using OpenCV Version = ', cv2.__version__)
print()

#colours
purple = (165, 0, 120)
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)

# Define string variable for path to file
strImage = "/Users/rachellucyshyn/Documents/GitHub/Vision2020-Competition/PowerCellImages/cell-03.png"

# load color image with string
imgBGRInput = cv2.imread(strImage)

# display color image to screen
cv2.imshow('Original Image', imgBGRInput) #window-title= what the window says at top

# Convert BGR to HSV
imgHSVinput = cv2.cvtColor(imgBGRInput, cv2.COLOR_BGR2HSV)

# Define range of colour in HSV (colour wheel- hue divide by 2 cause python is weird)
lower_yellow = np.array([10,100,100]) #hue/saturation/value (how much black or white)
upper_yellow = np.array([40,255,255]) # 255= zero black zero white

# Threshold the HSV image to get only yellow colors
imgBinaryMask = cv2.inRange(imgHSVinput, lower_yellow, upper_yellow)

# Bitwise-AND mask and original image
imgColorMask = cv2.bitwise_and(imgHSVinput,imgHSVinput, mask = imgBinaryMask) # frame = OG image


# display masked images
#cv2.imshow('Rebecca',imgHSVinput)
cv2.imshow('Binary Mask',imgBinaryMask)
cv2.imshow('ColorMask',imgColorMask)


#generate contours and display
imgFindContours, contours, hierarchy = cv2.findContours(imgBinaryMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
imgShowMaths = imgBGRInput.copy()
cv2.drawContours(imgBGRInput, contours, -1, (0,255,0), 20)# last parameter is width of line (0,255,0) is the colour greem
print('found contours = ', len(contours), 'contours in image')


cv2.imshow('imgBGRInput', imgBGRInput)

#sort contours by area descending
initialSortedContours = sorted(contours, key=cv2.contourArea, reverse = True)[:12] #reverse=order largest to smallest :12=largest 12


#calculate the moments and centroid
cnt = contours[0]
print('original contour length = ', len(cnt))

M = cv2.moments(cnt) #moments = help calculate some features
#print( M )

cx = 0
cy = 0

if abs(M['m00']) > 0:
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    print('centroid = ',cx,cy)

#parameters:begin coords, end coords, colour, width (from moments)
cv2.line(imgShowMaths, (cx-10,cy-10), (cx+10, cy+10), (0,255,0),2) #draw lines from opposite corners
cv2.line(imgShowMaths, (cx-10,cy+10), (cx+10, cy-10), (0,255,0),2)

cv2.drawContours(imgShowMaths, cnt, -1, purple, 10)

cv2.imshow('imgShowMaths', imgShowMaths)

#area
area = cv2.contourArea(cnt)
print('area = ', area) #in square pixels

#perimeter
perimeter = cv2.arcLength(cnt,True)
print('perimeter = ', perimeter)


#trace a shape with jagged-so it can be detected as a square (target reconstruction)
epsilon = 0.005*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)
#print('approx', approx)
#cv2.drawContours(imgShowMaths, approx, -1, (255,0,0), 7)
print('approx contour length = ', len(approx))
#cv2.imshow('approx over OG image', imgShowMaths) #put dots at corners


#convex hull=perimeter of simplified object-checks a curve for convexity defects and corrects it
hull = cv2.convexHull(cnt)
#print('hull', hull)
print('hull contour length = ', len(hull))
cv2.drawContours(imgShowMaths, hull, -1, red, 10)
hull_area = cv2.contourArea(hull)

if (hull_area) > 0:
    print('solidity from convex hull', float(area)/hull_area)

# Check Convexity
print('convexity is', cv2.isContourConvex(cnt))


#straight bounding rectangle
x,y,w,h = cv2.boundingRect(cnt)
print('straight bounding rectangle =', (x,y), w,h)
cv2.rectangle(imgShowMaths,(x,y),(x+w,y+h),green,2)
print('bounding rectangle aspect = ', float(w)/h) #make sure w is floating and result is floating
print('bounding rectangle extend = ', float(area)/(w*h))

#truncate=get rid of (if it wasn't a float, the decimals would be truncated)

 
#rotated rectangle
rect = cv2.minAreaRect(cnt)
print('rotated rectangle = ',rect)
(x,y),(width,height), angleofrotation = rect
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(imgShowMaths,[box],0,blue,2)

if height > 0:
    print('minimum area rectangle aspect = ', float(width)/height)
    print('minimum area rectangle extent = ', float(area)/(width*height))

# minimum enclosing circle
(x,y), radius = cv2.minEnclosingCircle(cnt)
print('minimum enclosing circle = ', (x,y),radius)
center = (int(x),int(y))
radius = int(radius)
cv2.circle(imgShowMaths,center,radius,green,2)
equi_diameter = np.sqrt(4*area/np.pi)
cv2.circle(imgShowMaths, (cx,cy), int(equi_diameter/2), purple, 3)


# fitting an elipse
ellipse = cv2.fitEllipse(cnt)
#print(ellipse)
# search ellipse to find it return a rotated rectangle in which the ellipse fits
(x,y),(majAxis,minAxis),angleofrotation = ellipse
print('ellipse center, maj axis, min axis, rotation = ', (x,y) ,(majAxis, minAxis), angleofrotation)
# search major and minor axis from ellipse
# https://namkeenman.wordpress.com/2015/12/21/opencv-determine-orientation-of-ellipserotatedrect-in-fitellipse/
cv2.ellipse(imgShowMaths,ellipse,red,2)
print('ellipse aspect = ', float(majAxis)/minAxis)

# fitting a line
rows,cols = imgBinaryMask.shape[:2]
#[vx,vy,x,y] = cv.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01) #errors in VS Code, search online and found fix
[vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
cv2.line(imgShowMaths,(cols-1,righty),(0,lefty),green,2)
# http://ottonello.gitlab.io/selfdriving/nanodegree/python/line%20detection/2016/12/18/extrapolating_lines.html
slope = vy / vx
intercept = y - (slope * x)
print('fitLine y = ', slope, '* x + ', intercept)

# Maximum Value, Minimum Value and their locations of a binary mask not contour!
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(imgBinaryMask)
print('min_val = ', min_val)
print('max_val = ', max_val)
print('min_loc = ', min_loc)
print('max_loc = ', max_loc)


# Mean Color or Mean Intensity 
mean_val1 = cv2.mean(imgBGRInput)
print('mean value from input image = ', mean_val1)
mean_val2 = cv2.mean(imgHSVinput, mask = imgBinaryMask)
print('mean value from HSV and mask = ', mean_val2)
# look at the result of mean_val2 on colorizer.org
mean_val3 = cv2.mean(imgColorMask)
print('mean value from colored mask = ', mean_val3)

# extreme points
leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
# draw extreme points
# from https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
cv2.circle(imgShowMaths, leftmost, 12, red, -1)
cv2.circle(imgShowMaths, rightmost, 12, green, -1)
cv2.circle(imgShowMaths, topmost, 12, (255, 255, 255), -1)
cv2.circle(imgShowMaths, bottommost, 12, blue, -1)
print('extreme points', leftmost,rightmost,topmost,bottommost)

# Display the contours and maths generated
cv2.imshow('contours and math over yellow mask', imgShowMaths)


# wait for user input to close
k = cv2.waitKey(0) #will close when key is pressed

# clean and exit
cv2.destroyAllWindows()