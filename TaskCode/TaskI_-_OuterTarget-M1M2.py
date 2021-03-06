# This is a pseudo code file for Merge Robotics, 2020, Infinite Recharge
# This is task I - > OpenCV "Contours Continued."  Not sure if it is clear by now, 
# but OpenCV can do a lot of things, we need to understand what it offers to complete 
# our vision code.  For a given single contour, (meaning it was imaged and masked and 
# converted to a coordinate array), you need to be able to use a number of OpenCV functions.
# Please experiment with the following, easiest is to simply draw them back to a blank image
# or on top of original.

# - contour perimeter, contour approximation, bounding rectangles, 
# minimum enclosing circle, fitting elipse, fitting line, aspect ratio
# extent, solidity, equivalent diameter, orientation, points, min/max
# mean color, extreme points

# useful links
# https://docs.opencv.org/3.4.7/dd/d49/tutorial_py_contour_features.html
# https://docs.opencv.org/3.4.7/d1/d32/tutorial_py_contour_properties.html

import numpy as np
import cv2
import sys
from pathlib import Path
import os
import math

print("Using python version {0}".format(sys.version))
print('OpenCV Version = ', cv2.__version__)
print()

# define colors
purple = (165, 0, 120)
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
cyan = (252, 252, 3)
magenta = (252, 3, 252)
yellow = (3, 252, 252)
black = (0, 0, 0)
white = (252, 252, 252)
orange = (3, 64, 252) 

# from https://stackoverflow.com/questions/41462419/python-slope-given-two-points-find-the-slope-answer-works-doesnt-work/41462583
def get_slope(x1, y1, x2, y2):
    return (y2-y1)/(x2-x1) 

# homemade! http://home.windstream.net/okrebs/Ch9-1.gif
def get_opposit(hyp, theta):
    return hyp*math.sin(math.radians(theta))

# homemade! http://home.windstream.net/okrebs/Ch9-1.gif
def get_adjacent(hyp, theta):
    return abs(hyp*math.cos(math.radians(theta)))

# select folder of interest
posCodePath = Path(__file__).absolute()
strVisionRoot = posCodePath.parent.parent
strImageFolder = str(strVisionRoot / 'OuterTargetHalfDistance')
#strImageFolder = str(strVisionRoot / 'OuterTargetSketchup')
#strImageFolder = str(strVisionRoot / 'OuterTargetHalfScale')
#strImageFolder = str(strVisionRoot / 'OuterTargetImages')

print (strImageFolder)
booBlankUpper = False

# read file names, and filter file names
photos = []
if os.path.exists(strImageFolder):
    for file in sorted(os.listdir(strImageFolder)):
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".JPG") or file.endswith(".PNG"):
            photos.append(file)
else:
    print
    print ('Directory', strImageFolder, 'does not exist, exiting ...')
    print
    sys.exit
print (photos)

# set index of files
i = 0
intLastFile = len(photos) -1

# begin main loop indent 1
while (True):

    ## set image input to indexed list
    strImageInput = strImageFolder + '/' + photos[i]
    ##print (i, ' ', strImageInput)
    print ()
    print (photos[i])

    ## read file
    imgImageInput = cv2.imread(strImageInput)
    intBinaryHeight, intBinaryWidth = imgImageInput.shape[:2]

    #cv2.imshow('imgImageInput', imgImageInput)
    #cv2.moveWindow('imgImageInput',300,350)

    # Convert BGR to HSV
    hsvImageInput = cv2.cvtColor(imgImageInput, cv2.COLOR_BGR2HSV)

    # define range of green color in HSV
    lower_color = np.array([55,45,40]) #np.array([65,80,40])
    upper_color = np.array([100,255,255])

    # Threshold the HSV image to get only green colors
    binary_mask = cv2.inRange(hsvImageInput, lower_color, upper_color)

    # Taking a matrix of size 5 as the kernel 
    #kernel = np.ones((3,3), np.uint8)
    #imgDilation = cv2.dilate(binary_mask, kernel, iterations=1)
    #binary_mask = imgDilation

    # mask the image to only show green or green images
    # Bitwise-AND mask and original image
    green_mask = cv2.bitwise_and(imgImageInput, imgImageInput, mask=binary_mask)

    # display the masked images to screen
    cv2.imshow('hsvImageInput', hsvImageInput)
    cv2.moveWindow('hsvImageInput',300,50)

    #cv2.imshow('binary_mask',binary_mask)
    cv2.imshow('green_masked',green_mask)
    #cv2.moveWindow('green_masked',400,550)
    cv2.moveWindow('green_masked',20,50) 
      
    # generate the contours and display
    imgFindContourReturn, contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    imgContours = green_mask.copy()

    cv2.drawContours(imgContours, contours, -1, yellow, 2)
    print('Found ', len(contours), 'contours in image')
    #print (contours)

    # sort contours by area descending
    initialSortedContours = sorted(contours, key = cv2.contourArea, reverse = True)[:3]

    if initialSortedContours:

        # Goind to work with largest area contour only
        cnt = initialSortedContours[0]
        print('original contour length = ', len(cnt))
        cv2.drawContours(imgContours, [cnt], -1, purple, 5)

        # Area
        area = cv2.contourArea(cnt)
        print('area = ', area)

        # Perimeter
        perimeter = cv2.arcLength(cnt,True)
        print('perimeter = ', perimeter)

        # Hull
        hull = cv2.convexHull(cnt)
        #print('hull', hull)
        print('hull contour length = ', len(hull))
        #cv2.drawContours(imgContours, [hull], -1, orange, 5)
        #cv2.imshow('hull over green mask', imgContours)
        hull_area = cv2.contourArea(hull)
        print('area of convex hull',hull_area)
        print('solidity from convex hull', float(area)/hull_area)

        # Check Convexity
        print('convexity is', cv2.isContourConvex(cnt))

        # straight bounding rectangle
        x,y,w,h = cv2.boundingRect(cnt)
        print('straight bounding rectangle = ', (x,y) ,w,h)
        #cv2.rectangle(imgContours,(x,y),(x+w,y+h),green,2)
        print('bounding rectangle aspect = ', float(w)/float(h))
        print('bounding rectangle extend = ', float(area)/(float(w)*float(h)))

        # Moment and Centroid
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            print('centroid = ',cx,cy)
            ct = int(h/12)
            cv2.line(imgContours,(cx-ct,cy-ct),(cx+ct,cy+ct),red,2)
            cv2.line(imgContours,(cx-ct,cy+ct),(cx+ct,cy-ct),red,2)

        # rotated rectangle
        rect = cv2.minAreaRect(cnt)
        print('rotated rectangle = ',rect)
        (x,y),(width,height),angleofrotation = rect
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        #cv2.drawContours(imgContours,[box],0,blue,2)
        minARaspect = float(width)/height
        print('minimum area rectangle aspect = ', minARaspect)
        print('minimum area rectangle extent = ', float(area)/(width*height))

        # minimum enclosing circle
        (xcirc,ycirc),radius = cv2.minEnclosingCircle(cnt)
        print('minimum enclosing circle = ', (xcirc,ycirc),radius)
        center = (int(xcirc),int(ycirc))
        radius = int(radius)
        cv2.circle(imgContours,center,radius,green,2)
        #equi_diameter = np.sqrt(4*area/np.pi)
        #cv2.circle(imgContours, (cx,cy), int(equi_diameter/2), purple, 3)

        if len(cnt) > 5:

            # fitting an elipse
            ellipse = cv2.fitEllipse(cnt)
            #print(ellipse)
            # search ellipse to find it return a rotated rectangle in which the ellipse fits
            (x,y),(majAxis,minAxis),angleofrotation = ellipse
            print('ellipse center, maj axis, min axis, rotation = ', (x,y) ,(majAxis, minAxis), angleofrotation)
            # search major and minor axis from ellipse
            # https://namkeenman.wordpress.com/2015/12/21/opencv-determine-orientation-of-ellipserotatedrect-in-fitellipse/
            #cv2.ellipse(imgContours,ellipse,red,2)
            print('ellipse aspect = ', float(majAxis)/minAxis)

        # fitting a line
        rows,cols = binary_mask.shape[:2]
        #[vx,vy,x,y] = cv.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01) #errors in VS Code, search online and found fix
        [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)
        #cv2.line(imgContours,(cols-1,righty),(0,lefty),green,2)
        # http://ottonello.gitlab.io/selfdriving/nanodegree/python/line%20detection/2016/12/18/extrapolating_lines.html
        slope = vy / vx
        intercept = y - (slope * x)
        print('fitLine y = ', slope, '* x + ', intercept)

        # aspect ratio
        # added retroactively to bounding, min area and elipse

        # extent calculation
        # added retroactively to bounding and min area

        # solidity
        # added retroactively to the hull

        # equivalent diameter
        # added retroactively to the enclosing circle

        # orientation
        # tweaked ellipse above to reflect details in link

        # mask and pixel points
        # skipping this one...

        # Maximum Value, Minimum Value and their locations of a binary mask not contour!
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(binary_mask)
        print('min_val = ', min_val)
        print('max_val = ', max_val)
        print('min_loc = ', min_loc)
        print('max_loc = ', max_loc)

        # Mean Color or Mean Intensity 
        mean_val1 = cv2.mean(imgImageInput)
        print('mean value from input image = ', mean_val1)
        mean_val2 = cv2.mean(hsvImageInput, mask = binary_mask)
        print('mean value from HSV and mask = ', mean_val2)
        # look at the result of mean_val2 on colorizer.org
        mean_val3 = cv2.mean(green_mask)
        print('mean value from colored mask = ', mean_val3)

        # extreme points
        leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
        rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
        #topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
        #bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
        # draw extreme points
        # from https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
        cv2.circle(imgContours, leftmost, 12, green, -1)
        cv2.circle(imgContours, rightmost, 12, red, -1)
        #cv2.circle(imgContours, topmost, 12, white, -1)
        #cv2.circle(imgContours, bottommost, 12, blue, -1)
        #print('extreme points = left',leftmost,'right',rightmost,'top',topmost,'bottom',bottommost)
        print('extreme points = left',leftmost,'right',rightmost)

        print('--')
        print (photos[i])
        print('minimum area rectangle aspect = ', minARaspect)
        print('leftmost minus rightmost -> ', int(rightmost[0])-int(leftmost[0]))
        print('minimum enclosing circle -> center / diameter = ', (xcirc,ycirc), radius*2)

    # Display the contours and maths generated
    cv2.imshow('contours and math over green mask', imgContours)
    cv2.moveWindow('contours and math over green mask',720,50)

    ## loop for user input to close - loop indent 2
    booReqToExit = False # true when user wants to exit

    while (True):

    ## wait for user to press key
        k = cv2.waitKey(0)
        if k == 27:
            booReqToExit = True # user wants to exit
            break
        elif k == 82: # user wants to move down list
            if i - 1 < 0:
                i = intLastFile
            else:
                i = i - 1
            break
        elif k == 84: # user wants to move up list
            if i + 1 > intLastFile:
                i = 0
            else:
                i = i + 1
            break
        elif k == 115:
            intMaskMethod = 0
            print()
            print('Mask Method s = Simple In-Range')
            break
        elif k == 107:
            intMaskMethod = 1
            print()
            print('Mask Method k = Knoxville Method')
            break
        elif k == 109:
            intMaskMethod = 2
            print()
            print('Mask Method m = Merge Mystery Method')
            break
        elif k == 32:
            print()
            print('...repeat...')
            break
        else:
            #print (k)
            pass
        ### end of loop indent 2

    ## test for exit main loop request from user
    if booReqToExit:
        break

    ## not exiting, close windows before loading next
    cv2.destroyAllWindows()

# end of main loop indent 1

# cleanup and exit
cv2.destroyAllWindows()