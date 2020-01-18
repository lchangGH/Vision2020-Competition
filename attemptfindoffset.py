# useful links
# https://docs.opencv.org/3.4.7/dd/d49/tutorial_py_contour_features.html
# https://docs.opencv.org/3.4.7/d1/d32/tutorial_py_contour_properties.html

import numpy as np
import cv2
import sys
from pathlib import Path
import glob

print("Using python version {0}".format(sys.version))
print('OpenCV Version = ', cv2.__version__)
print()

# define colors
purple = (165, 0, 120)
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
cyan = (252, 252, 3)
white = (255, 255, 255)


strImageFolder = "/Users/rachellucyshyn/Documents/GitHub/Vision2020-Competition/OuterTargetImages/"
print(strImageFolder)

# read and filter file names
photos1 = glob.glob(strImageFolder + '*.[jJ][pP][gG]')
photos2 = glob.glob(strImageFolder + '*.[jJ][pP][Ee][gG]')
photos3 = glob.glob(strImageFolder + '*.[pP][Nn][gG]')
photos = photos1 + photos2 + photos3


# set index of files
i = 0
intLastFile = len(photos) - 1

while(True):

    # load a color image using string
    imgImageInput = cv2.imread(photos[i])

    # display the color image to screen
    #cv2.imshow('input-image-title-bar', imgImageInput)

    # Convert BGR to HSV
    hsvImageInput = cv2.cvtColor(imgImageInput, cv2.COLOR_BGR2HSV)

    #need this to pick out target in "far protected zone" image
    #settingcolour
    lower_green = np.array([40,150,150])
    upper_green = np.array([80,255,255])

    # Threshold the HSV image to get only yellow colors
    binary_mask = cv2.inRange(hsvImageInput, lower_green, upper_green)

    # mask the image to only show yellow or green images
    # Bitwise-AND mask and original image
    colour_mask = cv2.bitwise_and(imgImageInput, imgImageInput, mask=binary_mask)

    # display the masked images to screen
    #cv2.imshow('hsvImageInput', hsvImageInput)
    #cv2.imshow('binary_mask',binary_mask)
    #cv2.imshow('colour_masked',colour_mask)

    # generate the contours and display
    imgFindContour, contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    imgContours = colour_mask.copy()
    cv2.drawContours(imgContours, contours, -1, purple, 10)
    print('Found ', len(contours), 'contours in image')

    #-------------------------
    # Need to determine the contour with the greatest height here

    if len(contours) == 0:
        print("No contours found")
        cv2.waitKey(0)  
            
    ### sort contours by area descending, keep only largest
    areaSortedContours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    print('Filtered to ', len(areaSortedContours), 'contours by area')

    ### draw the top contours in thin green
    cv2.drawContours(imgContours, areaSortedContours, -1, green, 3)

    ### create a holder or array for contours we want to keep in first filter
    tallestValidContour = []
    tallestRectangle = []
    floMaximumHeight = 0.0
    floWidthAtMaxHeight = 0.0
    floAngleAtMaxHeight = 0.0
    intIndexMaximumHeight = -1
        
    ### loop through area sorted contours, j is index, indiv is single contour
    for (j, indiv) in enumerate(areaSortedContours):

        #### determine minimum area rectangle
        rectangle = cv2.minAreaRect(indiv)
        (xm,ym),(wm,hm), am = rectangle

        #### print to console
        if abs(hm) > 0:
            print ('index=',j,'height=',hm,'width=',wm,'angle=',am,'minAreaAspect=',wm/hm)
        else:
            print ('index=',j,'height=',hm,'width=',wm,'angle=',am)

        #### track tallest contour that looks like a cube based on extent
        if (hm > floMaximumHeight):
            floMaxHtMinaX = xm
            floMaxHtMinaY = ym
            floMaximumHeight = hm
            floWidthAtMaxHeight = wm
            floAngleAtMaxHeight = am
            intIndexMaximumHeight = j
            tallestRectangle = rectangle

    # taking the tallest contour, do some calculations for direction finding
    if intIndexMaximumHeight == -1: # 0 or higher means a valid tallest contour found
        break

    cnt = areaSortedContours[intIndexMaximumHeight]

    #-------------------------

    # Moment and Centroid
    # RL took next line out
    #cnt = contours[0]
    #print(cnt)
    #print('original',len(cnt),cnt)
    print('original contour length = ', len(cnt))
    M = cv2.moments(cnt)
    #print( M )
    cx = 0
    cy = 0
    if abs(M['m00']) > 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        print('centroid = ',cx,cy)
    cv2.line(imgContours,(cx-10,cy-10),(cx+10,cy+10),red,2)
    cv2.line(imgContours,(cx-10,cy+10),(cx+10,cy-10),red,2)

    cv2.drawContours(imgContours, cnt, -1, purple, 10)

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
    cv2.drawContours(imgContours, hull, -1, red, 10)
    #cv2.imshow('hull over yellow mask', imgContours)
    hull_area = cv2.contourArea(hull)
    if abs(hull_area) > 0:
        print('solidity from convex hull', float(area)/hull_area)

    # Check Convexity
    print('convexity is', cv2.isContourConvex(cnt))

    # straight bounding rectangle
    x,y,w,h = cv2.boundingRect(cnt)
    print('straight bounding rectangle = ', (x,y) ,w,h)
    cv2.rectangle(imgContours,(x,y),(x+w,y+h),green,2)
    print('bounding rectangle aspect = ', float(w)/float(h))
    print('bounding rectangle extend = ', float(area)/(float(w)*float(h)))

    # rotated rectangle
    rect = cv2.minAreaRect(cnt)
    print('rotated rectangle = ',rect)
    (x,y),(width,height),angleofrotation = rect
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(imgContours,[box],0,blue,2)
    if abs(height) > 0:
        print('minimum area rectangle aspect = ', float(width)/height)
        print('minimum area rectangle extent = ', float(area)/(width*height))

        #----------------------------------------
        # calculate offset of middle of bounding box from center
        midBoundingRect = (leftside + rightside)/2
        


    # minimum enclosing circle
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    print('minimum enclosing circle = ', (x,y),radius)
    center = (int(x),int(y))
    radius = int(radius)
    cv2.circle(imgContours,center,radius,green,2)
    equi_diameter = np.sqrt(4*area/np.pi)
    cv2.circle(imgContours, (cx,cy), int(equi_diameter/2), purple, 3)

    # fitting an elipse
    if len(cnt) > 5:
        ellipse = cv2.fitEllipse(cnt)
        #print(ellipse)
        # search ellipse to find it return a rotated rectangle in which the ellipse fits
        (x,y),(majAxis,minAxis),angleofrotation = ellipse
        print('ellipse center, maj axis, min axis, rotation = ', (x,y) ,(majAxis, minAxis), angleofrotation)
        # search major and minor axis from ellipse
        # https://namkeenman.wordpress.com/2015/12/21/opencv-determine-orientation-of-ellipserotatedrect-in-fitellipse/
        cv2.ellipse(imgContours,ellipse,red,2)
        print('ellipse aspect = ', float(majAxis)/minAxis)

    # fitting a line
    rows,cols = binary_mask.shape[:2]
    #[vx,vy,x,y] = cv.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01) #errors in VS Code, search online and found fix
    [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    cv2.line(imgContours,(cols-1,righty),(0,lefty),green,2)
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
    mean_val3 = cv2.mean(colour_mask)
    print('mean value from colored mask = ', mean_val3)

    # extreme points
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
    # draw extreme points
    # from https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
    cv2.circle(imgContours, leftmost, 8, green, -1)
    cv2.circle(imgContours, rightmost, 8, red, -1)
    cv2.circle(imgContours, topmost, 8, white, -1)
    cv2.circle(imgContours, bottommost, 8, blue, -1)
    print('extreme points', leftmost,rightmost,topmost,bottommost)


    # Display the contours and maths generated
    cv2.imshow('contours and math over colour mask', imgContours)

    ## loop for user input to close - loop indent 2
    booReqToExit = False # true when user wants to exit

    while (True):

        ### wait for user to press key
        k = cv2.waitKey(0)
        if k == 27:
            booReqToExit = True # user wants to exit
            break
        elif k == 97: # 'a' user wants to move down list
            if i - 1 < 0:
                i = intLastFile
            else:
                i = i - 1
            break
        elif k == 122: # 'z' user wants to move up list
            if i + 1 > intLastFile:
                i = 0
            else:
                i = i + 1
            break

        else:
            print (k)

        ### end of loop indent 2

    ## test for exit main loop request from user
    if booReqToExit:
        break

    ## not exiting, close windows before loading next
    cv2.destroyAllWindows()