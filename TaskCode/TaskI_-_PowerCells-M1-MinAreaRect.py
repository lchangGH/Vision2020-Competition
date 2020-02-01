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

# select folder of interest
posCodePath = Path(__file__).absolute()
strVisionRoot = posCodePath.parent.parent
#strImageFolder = str(strVisionRoot / 'CalibrationImages')
#strImageFolder = str(strVisionRoot / 'ProblemImages')
#strImageFolder = str(strVisionRoot / 'DistanceImages') 
#strImageFolder = str(strVisionRoot / '2706-Elimins-Images')
#strImageFolder = str(strVisionRoot / 'PowerCellImages')
#strImageFolder = str(strVisionRoot / 'PowerCell25Scale')
strImageFolder = str(strVisionRoot / 'PowerCellFullScale')
#strImageFolder = str(strVisionRoot / 'PowerCellFullMystery')
strImageFolder = str(strVisionRoot / 'PowerCellSketchup')

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
i = 10
intLastFile = len(photos) -1

# begin main loop indent 1
while (True):

    cv2.destroyAllWindows()

    ## start timer for loop as indication of FPS...
    ##floStartTimeA = time.perf_counter()
    ##print ('start = ',floStartTimeA)

    ## set image input to indexed list
    strImageInput = strImageFolder + '/' + photos[i]
    ##print (i, ' ', strImageInput)
    print ()
    print (photos[i])

    ## read file
    imgImageInput = cv2.imread(strImageInput)

    if booBlankUpper:
        ## blank upper portion from Task K
        intBinaryHeight,intBinaryWidth = imgImageInput.shape[:2]
        cv2.rectangle(imgImageInput, (0,0), (intBinaryWidth, int(intBinaryHeight/2-10)), black, -1)

    # Convert BGR to HSV
    hsvImageInput = cv2.cvtColor(imgImageInput, cv2.COLOR_BGR2HSV)

    # define range of yellow color in HSV
    lower_color = np.array([20,65,80])
    upper_color = np.array([65,255,255])

    # Threshold the HSV image to get only yellow colors
    binary_mask = cv2.inRange(hsvImageInput, lower_color, upper_color)

    # mask the image to only show yellow or green images
    # Bitwise-AND mask and original image
    yellow_mask = cv2.bitwise_and(imgImageInput, imgImageInput, mask=binary_mask)

    # display the masked images to screen
    cv2.imshow('hsvImageInput', hsvImageInput)
    cv2.moveWindow('hsvImageInput',100,50)

    #cv2.imshow('binary_mask',binary_mask)

    cv2.imshow('yellow_masked',yellow_mask)
    cv2.moveWindow('yellow_masked',200,650)

    ### wait for user to press any key
    #while (True):
    #    k = cv2.waitKey(0)
    #    break

    # generate the contours and display
    imgFindContourReturn, contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    imgContours = yellow_mask.copy()

    cv2.drawContours(imgContours, contours, -1, cyan, 2)
    print('Found ', len(contours), 'contours in image')
    #print (contours)

    # sort contours by area descending
    initialSortedContours = sorted(contours, key = cv2.contourArea, reverse = True)[:3]

    # Moment and Centroid
    cnt = initialSortedContours[0]
    #print(cnt)
    #print('original',len(cnt),cnt)
    print('original contour length = ', len(cnt))
    M = cv2.moments(cnt)
    #print( M )

    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    print('centroid = ',cx,cy)
    cv2.line(imgContours,(cx-10,cy-10),(cx+10,cy+10),red,2)
    cv2.line(imgContours,(cx-10,cy+10),(cx+10,cy-10),red,2)

    cv2.drawContours(imgContours, cnt, -1, purple, 7)

    # Area
    area = cv2.contourArea(cnt)
    print('area = ', area)

    # Perimeter
    perimeter = cv2.arcLength(cnt,True)
    print('perimeter = ', perimeter)

    # Contour Approximation
    epsilon = 0.005*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    #print('approx', approx)
    #cv2.drawContours(imgContours, approx, -1, red, 10)
    print('approx contour length = ', len(approx))
    #cv2.imshow('approx over yellow mask', imgContours)

    # Hull
    hull = cv2.convexHull(cnt)
    #print('hull', hull)
    print('hull contour length = ', len(hull))
    #cv2.drawContours(imgContours, hull, -1, red, 10)
    #cv2.imshow('hull over yellow mask', imgContours)
    hull_area = cv2.contourArea(hull)
    print('area from convex hull', float(hull_area))
    print('solidity from convex hull', float(area)/hull_area)

    # Check Convexity
    print('convexity is', cv2.isContourConvex(cnt))

    # straight bounding rectangle
    x,y,w,h = cv2.boundingRect(cnt)
    print('straight bounding rectangle = ', (x,y) ,w,h)
    #cv2.rectangle(imgContours,(x,y),(x+w,y+h),green,2)
    print('bounding rectangle aspect = ', float(w)/float(h))
    print('bounding rectangle extend = ', float(area)/(float(w)*float(h)))

    # rotated rectangle
    rect = cv2.minAreaRect(cnt)
    print('rotated rectangle = ',rect)
    (x,y),(width,height),angleofrotation = rect
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(imgContours,[box],0,blue,2)
    print('minimum area rectangle aspect = ', float(width)/height)
    print('minimum area rectangle extent = ', float(area)/(width*height))

    # minimum enclosing circle
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    print('minimum enclosing circle = ', (x,y),radius)
    center = (int(x),int(y))
    radius = int(radius)
    #cv2.circle(imgContours,center,radius,green,2)
    equi_diameter = np.sqrt(4*area/np.pi)
    #cv2.circle(imgContours, (cx,cy), int(equi_diameter/2), purple, 3)

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
    mean_val3 = cv2.mean(yellow_mask)
    print('mean value from colored mask = ', mean_val3)

    # extreme points
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
    # draw extreme points
    # from https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
    #cv2.circle(imgContours, leftmost, 12, green, -1)
    #cv2.circle(imgContours, rightmost, 12, red, -1)
    #cv2.circle(imgContours, topmost, 12, white, -1)
    #cv2.circle(imgContours, bottommost, 12, blue, -1)
    print('extreme points = left',leftmost,'right',rightmost,'top',topmost,'bottom',bottommost)

    # Display the contours and maths generated
    cv2.imshow('contours and math over yellow mask', imgContours)
    cv2.moveWindow('contours and math over yellow mask',600,500)

    ## loop for user input to close - loop indent 2
    booReqToExit = False # true when user wants to exit

    while (True):

        ### wait for user to press key
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
        elif k == 119:
            pass

        else:
            print (k)
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