# ----------------------------------------------------------------------------
# Copyright (c) 2018 FIRST. All Rights Reserved.
# Open Source Software - may be modified and shared by FRC teams. The code
# must be accompanied by the FIRST BSD license file in the root directory of
# the project.

# My 2020 license: use it as much as you want. Crediting is recommended because it lets me know 
# that I am being useful.
# Some parts of pipeline are based on 2019 code created by the Screaming Chickens 3997

# This is meant to be used in conjuction with WPILib Raspberry Pi image: https://github.com/wpilibsuite/FRCVision-pi-gen
# ----------------------------------------------------------------------------

import json
import time
import sys
from threading import Thread
import random


import cv2
import numpy as np

import math

import os

########### SET RESOLUTION TO 256x144 !!!! ############

# import the necessary packages
import datetime


# Class to examine Frames per second of camera stream. Currently not used.


###################### PROCESSING OPENCV ################################

# counts frames for writing images
frameStop = 0
ImageCounter = 0

# Angles in radians

# image size ratioed to 16:9


# Lifecam 3000 from datasheet
# Datasheet: https://dl2jx7zfbtwvr.cloudfront.net/specsheets/WEBC1010.pdf

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

#images = load_images_from_folder("./OuterTargetImages")
#images = load_images_from_folder("./OuterTargetHalfScale")
#images = load_images_from_folder("./PowerCell25Scale")

#mages = load_images_from_folder("./PowerCellImages")
images = load_images_from_folder("./PowerCellFullScale")

# finds height/width of camera frame (eg. 640 width, 480 height)
image_height, image_width = images[0].shape[:2]
print(image_height, image_width)

# FOV of microsoft camera (68.5 is camera spec)
diagonalView = math.radians(68.5)

print("Diagonal View:" + str(diagonalView))

# 4:3 aspect ratio
horizontalAspect = 4
verticalAspect = 3

# Reasons for using diagonal aspect is to calculate horizontal field of view.
diagonalAspect = math.hypot(horizontalAspect, verticalAspect)
# Calculations: http://vrguy.blogspot.com/2013/04/converting-diagonal-field-of-view-and.html
horizontalView = math.atan(math.tan(diagonalView / 2) * (horizontalAspect / diagonalAspect)) * 2
verticalView = math.atan(math.tan(diagonalView / 2) * (verticalAspect / diagonalAspect)) * 2

# Focal Length calculations: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_165
H_FOCAL_LENGTH = image_width / (2 * math.tan((horizontalView / 2)))
V_FOCAL_LENGTH = image_height / (2 * math.tan((verticalView / 2)))
# blurs have to be odd
green_blur = 1
orange_blur = 27
yellow_blur = 27

# define range of green of retroreflective tape in HSV
lower_green = np.array([40, 75, 75])
upper_green = np.array([96, 255, 255])

# define range of yellow in HSV
lower_yellow = np.array([20, 25, 30])
upper_yellow = np.array([70, 255, 255])

# initialize some variable used later for user input
color_is_yellow = True
lower_color = lower_yellow
upper_color = upper_yellow


switch = 1


# Flip image if camera mounted upside down
def flipImage(frame):
    return cv2.flip(frame, -1)


# Blurs frame
def blurImg(frame, blur_radius):
    img = frame.copy()
    blur = cv2.blur(img, (blur_radius, blur_radius))
    return blur


def threshold_range(im, lo, hi):
    unused, t1 = cv2.threshold(im, lo, 255, type=cv2.THRESH_BINARY)
    unused, t2 = cv2.threshold(im, hi, 255, type=cv2.THRESH_BINARY_INV)
    return cv2.bitwise_and(t1, t2)


# Masks the video based on a range of hsv colors
# Takes in a frame, range of color, and a blurred frame, returns a masked frame
def threshold_video(lower_color, upper_color, blur):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    h = threshold_range(h, lower_color[0], upper_color[0])
    s = threshold_range(s, lower_color[1], upper_color[1])
    v = threshold_range(v, lower_color[2], upper_color[2])
    combined_mask = cv2.bitwise_and(h, cv2.bitwise_and(s, v))
    

    # hold the HSV image to get only red colors
    # mask = cv2.inRange(combined, lower_color, upper_color)

    # Returns the masked imageBlurs video to smooth out image

    return combined_mask


# Finds the tape targets from the masked image and displays them on original stream + network tales
def findTargets(frame, mask):


    # Finds contours
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    # Take each frame
    # Gets the shape of video
    screenHeight, screenWidth, _ = frame.shape
    # Gets center of height and width
    centerX = (screenWidth / 2) - .5
    centerY = (screenHeight / 2) - .5
    # Copies frame and stores it in image
    image = frame.copy()
    # Processes the contours, takes in (contours, output_image, (centerOfImage)
    if len(contours) != 0:
        #image = findTape(contours, image, centerX, centerY)
        image = findOuterTarget(contours, image, centerX, centerY)
    # Shows the contours overlayed on the original video
    return image


# Finds the balls from the masked image and displays them on original stream + network tables
def findPowerCell(frame, mask): 
    
    # Finds contours
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    # Take each frame
    # Gets the shape of video
    screenHeight, screenWidth, _ = frame.shape
    # Gets center of height and width
    centerX = (screenWidth / 2) - .5
    centerY = (screenHeight / 2) - .5
    # Copies frame and stores it in image
    image = frame.copy()
    # Processes the contours, takes in (contours, output_image, (centerOfImage)
    if len(contours) != 0:
        image = findBall(contours, image, centerX, centerY)
    # Shows the contours overlayed on the original video
    return image

#find outer target
def findOuterTarget(frame, mask): 

    # find the three points (leftmost, rightmost, bottomost)

    # Finds contours
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    # Take each frame
    # Gets the shape of video
    screenHeight, screenWidth, _ = frame.shape
    # Gets center of height and width
    centerX = (screenWidth / 2) - .5
    centerY = (screenHeight / 2) - .5
    # Copies frame and stores it in image
    image = frame.copy()
    # Processes the contours, takes in (contours, output_image, (centerOfImage)

    if len(contours) != 0:

        # Sort contours by area size (biggest to smallest)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        outerTargetContour = cntsSorted[0];

        #  to be completed

    # Call solvePnp()

    # Read Image
    im = cv2.imread("headPose.jpg");
    size = im.shape

    #2D image points. If you change the image, you need to change vector
    image_points = np.array([ (359, 391), # Nose tip
                              (399, 561), # Chin
                              (337, 297), # Left eye left corner
                              (513, 301), # Right eye right corner
                              (345, 465), # Left Mouth corner
                              (453, 469)  # Right mouth corner
                              ], dtype="double")

    # 3D model points.
    model_points = np.array([ (0.0, 0.0, 0.0), # Nose tip
                              (0.0, -330.0, -65.0), # Chin
                              (-225.0, 170.0, -135.0), # Left eye left corner
                              (225.0, 170.0, -135.0), # Right eye right corne
                              (-150.0, -150.0, -125.0), # Left Mouth corner
                              (150.0, -150.0, -125.0) # Right mouth corner
                              ])

    # Camera internals
    
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array( [[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]
                              ], dtype = "double")

    print ("Camera Matrix :\n {0}", format(camera_matrix))
    
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.CV_ITERATIVE)
    
    print ("Rotation Vector: ", format(rotation_vector))
    print ("Translation Vector:\n {0}", format(translation_vector))

    # Shows the contours overlayed on the original video
    return image


# Draws Contours and finds center and yaw of orange ball
# centerX is center x coordinate of image
# centerY is center y coordinate of image
def findBall(contours, image, centerX, centerY):
    screenHeight, screenWidth, channels = image.shape
    # Seen vision targets (correct angle, adjacent to each other)
    cargo = []

    if len(contours) > 0:
        # Sort contours by area size (biggest to smallest)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        cntHeight = 0
        biggestPowerCell = []
        for cnt in cntsSorted:
            x, y, w, h = cv2.boundingRect(cnt)
        
            cntHeight = h
            aspect_ratio = float(w) / h
            # Get moments of contour; mainly for centroid
            M = cv2.moments(cnt)
            # Get convex hull (bounding polygon on contour)
            hull = cv2.convexHull(cnt)
            # Calculate Contour area
            cntArea = cv2.contourArea(cnt)
            # Filters contours based off of size
            if (checkBall(cntArea, aspect_ratio)):
                ### MOSTLY DRAWING CODE, BUT CALCULATES IMPORTANT INFO ###
                # Gets the centeroids of contour
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                if (len(biggestPowerCell) < 3):

                    ##### DRAWS CONTOUR######
                    # Gets rotated bounding rectangle of contour
                    rect = cv2.minAreaRect(cnt)
                    # Creates box around that rectangle
                    box = cv2.boxPoints(rect)
                    # Not exactly sure
                    box = np.int0(box)
                    # Draws rotated rectangle
                    cv2.drawContours(image, [box], 0, (23, 184, 80), 3)

                    # Draws a vertical white line passing through center of contour
                    cv2.line(image, (cx, screenHeight), (cx, 0), (255, 255, 255))
                    # Draws a white circle at center of contour
                    cv2.circle(image, (cx, cy), 6, (255, 255, 255))

                    # Draws the contours
                    cv2.drawContours(image, [cnt], 0, (23, 184, 80), 1)

                    # Gets the (x, y) and radius of the enclosing circle of contour
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    # Rounds center of enclosing circle
                    center = (int(x), int(y))
                    # Rounds radius of enclosning circle
                    radius = int(radius)
                    # Makes bounding rectangle of contour
                    rx, ry, rw, rh = cv2.boundingRect(cnt)

                    # Draws countour of bounding rectangle and enclosing circle in green
                    cv2.rectangle(image, (rx, ry), (rx + rw, ry + rh), (23, 184, 80), 1)

                    cv2.circle(image, center, radius, (23, 184, 80), 1)

                    # Appends important info to array
                    if [cx, cy, cnt, cntHeight] not in biggestPowerCell:
                        biggestPowerCell.append([cx, cy, cnt, cntHeight])

        # Check if there are cargo seen
        if (len(biggestPowerCell) > 0):
            # pushes that it sees cargo to network tables

            finalTarget = []
            # Sorts targets based on x coords to break any angle tie
            biggestPowerCell.sort(key=lambda x: math.fabs(x[0]))
            closestCargo = min(biggestPowerCell, key=lambda x: (math.fabs(x[0] - centerX)))
            xCoord = closestCargo[0]
            finalTarget.append(calculateYaw(xCoord, centerX, H_FOCAL_LENGTH))
            finalTarget.append(calculateDistWPILib(closestCargo[3]))
            print("Yaw: " + str(finalTarget[0]))

            # Puts the yaw on screen
            # Draws yaw of target + line where center of target is
            cv2.putText(image, "Yaw: " + str(finalTarget[0]), (40, 40), cv2.FONT_HERSHEY_COMPLEX, .6,
                        (255, 255, 255))
            cv2.putText(image, "Dist: " + str(finalTarget[1]), (40, 100), cv2.FONT_HERSHEY_COMPLEX, .6,
                        (255, 255, 255))
            cv2.line(image, (xCoord, screenHeight), (xCoord, 0), (255, 0, 0), 2)

            currentAngleError = finalTarget[0]
            # pushes cargo angle to network tables



        cv2.line(image, (round(centerX), screenHeight), (round(centerX), 0), (255, 255, 255), 2)

        return image



# Draws Contours and finds center and yaw of vision targets
# centerX is center x coordinate of image
# centerY is center y coordinate of image
def findTape(contours, image, centerX, centerY):
    screenHeight, screenWidth, channels = image.shape
    # Seen vision targets (correct angle, adjacent to each other)
    targets = []

    if len(contours) >= 2:
        # Sort contours by area size (biggest to smallest)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        cntHeight = 0

        biggestCnts = []
        for cnt in cntsSorted:
            # Get moments of contour; mainly for centroid
            M = cv2.moments(cnt)
            # Get convex hull (bounding polygon on contour)
            hull = cv2.convexHull(cnt)
            # Calculate Contour area
            cntArea = cv2.contourArea(cnt)
            # calculate area of convex hull
            hullArea = cv2.contourArea(hull)

            x, y, w, cntHeight = cv2.boundingRect(cnt)

            pts, dim, a = cv2.minAreaRect(cnt)

            x = pts[0]
            y = pts[1]

            if dim[0] > dim[1]:
                cntHeight = dim[0]
            else:
                cntHeight = dim[1]

            # print("The contour height is, ", cntHeight)
            # Filters contours based off of size
            if (checkContours(cntArea, hullArea)):
                ### MOSTLY DRAWING CODE, BUT CALCULATES IMPORTANT INFO ###
                # Gets the centeroids of contour
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    theCX = cx
                    theCY = cy
                else:
                    cx, cy = 0, 0
                if (len(biggestCnts) < 13):
                    #### CALCULATES ROTATION OF CONTOUR BY FITTING ELLIPSE ##########
                    rotation = getEllipseRotation(image, cnt)

                    # Calculates yaw of contour (horizontal position in degrees)
                    yaw = calculateYaw(cx, centerX, H_FOCAL_LENGTH)
                    # Calculates yaw of contour (horizontal position in degrees)
                    pitch = calculatePitch(cy, centerY, V_FOCAL_LENGTH)
                    # Calculates Distance
                    dist = calculateDistance(1, 2, pitch);

                    ##### DRAWS CONTOUR######
                    # Gets rotated bounding rectangle of contour
                    rect = cv2.minAreaRect(cnt)
                    # Creates box around that rectangle
                    box = cv2.boxPoints(rect)
                    # Not exactly sure
                    box = np.int0(box)
                    # Draws rotated rectangle

                    cv2.drawContours(image, [box], 0, (23, 184, 80), 3)

                    # Calculates yaw of contour (horizontal position in degrees)
                    yaw = calculateYaw(cx, centerX, H_FOCAL_LENGTH)
                    # Calculates yaw of contour (horizontal position in degrees)
                    pitch = calculatePitch(cy, centerY, V_FOCAL_LENGTH)
                    # Calculates Distance
                    dist = calculateDistance(1, 2, pitch);

                    # Draws a vertical white line passing through center of contour
                    cv2.line(image, (cx, screenHeight), (cx, 0), (255, 255, 255))
                    # Draws a white circle at center of contour
                    cv2.circle(image, (cx, cy), 6, (255, 255, 255))

                    # Draws the contours
                    cv2.drawContours(image, [cnt], 0, (23, 184, 80), 1)

                    # Gets the (x, y) and radius of the enclosing circle of contour
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    # Rounds center of enclosing circle
                    center = (int(x), int(y))
                    # Rounds radius of enclosning circle
                    radius = int(radius)
                    # Makes bounding rectangle of contour
                    rx, ry, rw, rh = cv2.boundingRect(cnt)
                    boundingRect = cv2.boundingRect(cnt)
                    # Draws countour of bounding rectangle and enclosing circle in green
                    cv2.rectangle(image, (rx, ry), (rx + rw, ry + rh), (23, 184, 80), 1)

                    cv2.circle(image, center, radius, (23, 184, 80), 1)

                    # Appends important info to array
                    if [cx, cy, rotation, cnt, cntHeight] not in biggestCnts:
                        biggestCnts.append([cx, cy, rotation, cnt, cntHeight])

        # Sorts array based on coordinates (leftmost to rightmost) to make sure contours are adjacent
        biggestCnts = sorted(biggestCnts, key=lambda x: x[0])
        # Target Checking
        for i in range(len(biggestCnts) - 1):
            # Rotation of two adjacent contours
            tilt1 = biggestCnts[i][2]
            tilt2 = biggestCnts[i + 1][2]

            # x coords of contours
            cx1 = biggestCnts[i][0]
            cx2 = biggestCnts[i + 1][0]

            cy1 = biggestCnts[i][1]
            cy2 = biggestCnts[i + 1][1]
            # If contour angles are opposite
            if (np.sign(tilt1) != np.sign(tilt2)):
                centerOfTarget = math.floor((cx1 + cx2) / 2)
                # ellipse negative tilt means rotated to right
                # Note: if using rotated rect (min area rectangle)
                #      negative tilt means rotated to left
                # If left contour rotation is tilted to the left then skip iteration
                if (tilt1 > 0):
                    if (cx1 < cx2):
                        continue
                # If left contour rotation is tilted to the left then skip iteration
                if (tilt2 > 0):
                    if (cx2 < cx1):
                        continue
                # Angle from center of camera to target (what you should pass into gyro)
                yawToTarget = calculateYaw(centerOfTarget, centerX, H_FOCAL_LENGTH)
                pitchToTarget = calculatePitch(theCY, centerY, H_FOCAL_LENGTH)
                # distToTarget = calculateDistance(1, 2, pitchToTarget)
                distToTarget = calculateDistWPILib(biggestCnts[i][4])
                # Make sure no duplicates, then append
                if [centerOfTarget, yawToTarget, distToTarget] not in targets:
                    targets.append([centerOfTarget, yawToTarget, distToTarget])
    # Check if there are targets seen
    if (len(targets) > 0):
        # pushes that it sees vision target to network tables

        # Sorts targets based on x coords to break any angle tie
        targets.sort(key=lambda x: math.fabs(x[0]))
        finalTarget = min(targets, key=lambda x: math.fabs(x[1]))
        # Puts the yaw on screen
        # Draws yaw of target + line where center of target is
        cv2.putText(image, "Yaw: " + str(finalTarget[1]), (40, 40), cv2.FONT_HERSHEY_COMPLEX, .6,
                    (255, 255, 255))
        cv2.putText(image, "Dist: " + str(finalTarget[2]), (40, 90), cv2.FONT_HERSHEY_COMPLEX, .6,
                    (255, 255, 255))
        cv2.line(image, (finalTarget[0], screenHeight), (finalTarget[0], 0), (255, 0, 0), 2)

        currentAngleError = finalTarget[1]
        # pushes vision target angle to network table


    cv2.line(image, (round(centerX), screenHeight), (round(centerX), 0), (255, 255, 255), 2)

    return image


# Finds the balls from the masked image and displays them on original stream + network tables
def findControlPanel(frame, mask):
    # Finds contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    # Take each frame
    # Gets the shape of video
    screenHeight, screenWidth, _ = frame.shape
    # Gets center of height and width
    centerX = (screenWidth / 2) - .5
    centerY = (screenHeight / 2) - .5
    # Copies frame and stores it in image
    image = frame.copy()
    # Processes the contours, takes in (contours, output_image, (centerOfImage)
    if len(contours) != 0:
        image = findControlPanelColour(contours, image, centerX, centerY)
    # Shows the contours overlayed on the original video
    return image

# Draws Contours and finds the colour the control panel wheel is resting at
# centerX is center x coordinate of image
# centerY is center y coordinate of image
def findControlPanelColour(contours, image, centerX, centerY):
    #ToDo, Add code to publish wheel colour
    return image

# Checks if tape contours are worthy based off of contour area and (not currently) hull area
def checkContours(cntSize, hullSize):
    print(cntSize, image_width / 7)
    return cntSize > (image_width / 7)


# Checks if ball contours are worthy based off of contour area and (not currently) hull area
def checkBall(cntSize, cntAspectRatio):
    return (cntSize > (image_width / 2)) and (round(cntAspectRatio) == 1)


# Forgot how exactly it works, but it works!
def translateRotation(rotation, width, height):
    if (width < height):
        rotation = -1 * (rotation - 90)
    if (rotation > 90):
        rotation = -1 * (rotation - 180)
    rotation *= -1
    return round(rotation)


def calculateDistance(heightOfCamera, heightOfTarget, pitch):
    heightOfTargetFromCamera = heightOfTarget - heightOfCamera

    # Uses trig and pitch to find distance to target
    '''
    d = distance
    h = height between camera and target
    a = angle = pitch
    tan a = h/d (opposite over adjacent)
    d = h / tan a
                         .
                        /|
                       / |
                      /  |h
                     /a  |
              camera -----
                       d
    '''
    divisor = math.tan(math.radians(pitch))
    distance = 0
    if (divisor != 0):
        distance = math.fabs(heightOfTargetFromCamera / divisor)

    return distance


avg = [0 for i in range(0, 1)]
#8 is number of frames to calculated average pixel height

def calculateDistWPILib(cntHeight):
    global image_height, avg

    for cnt in avg:
        if cnt == 0:
            cnt = cntHeight

    del avg[len(avg) - 1]
    avg.insert(0, cntHeight)
    PIX_HEIGHT = 0
    for cnt in avg:
        PIX_HEIGHT += cnt

    PIX_HEIGHT = PIX_HEIGHT / len(avg)

    print (PIX_HEIGHT)



    print(PIX_HEIGHT, avg)  # print("The contour height is: ", cntHeight)

    #TARGET_HEIGHT is actual height (for balls 7/12 7 inches)   
    TARGET_HEIGHT = 0.583

 
    #image height is the y resolution calculated from image size
    #15.81 was the pixel height of a a ball found at a measured distance (which is 6 feet away)
    #65 is the pixel height of a scale image 6 feet away
    KNOWN_OBJECT_PIXEL_HEIGHT = 65
    KNOWN_OBJECT_DISTANCE = 6
    VIEWANGLE = math.atan((TARGET_HEIGHT * image_height) / (2 * KNOWN_OBJECT_PIXEL_HEIGHT * KNOWN_OBJECT_DISTANCE))

    # print("after 2: ", VIEWANGLE)
    # VIEWANGLE = math.radians(68.5)
    distance = ((TARGET_HEIGHT * image_height) / (2 * PIX_HEIGHT * math.tan(VIEWANGLE)))
    # distance = ((0.02) * distance ** 2) + ((69/ 100) * distance) + (47 / 50)
    # distance = ((-41/450) * distance ** 2) + ((149 / 100) * distance) - (9 / 25)

    return distance


# Uses trig and focal length of camera to find yaw.
# Link to further explanation: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298
def calculateYaw(pixelX, centerX, hFocalLength):
    yaw = math.degrees(math.atan((pixelX - centerX) / hFocalLength))
    return round(yaw)


# Link to further explanation: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298
def calculatePitch(pixelY, centerY, vFocalLength):
    pitch = math.degrees(math.atan((pixelY - centerY) / vFocalLength))
    # Just stopped working have to do this:
    pitch *= -1
    return round(pitch)


def getEllipseRotation(image, cnt):
    try:
        # Gets rotated bounding ellipse of contour
        ellipse = cv2.fitEllipse(cnt)
        centerE = ellipse[0]
        # Gets rotation of ellipse; same as rotation of contour
        rotation = ellipse[2]
        # Gets width and height of rotated ellipse
        widthE = ellipse[1][0]
        heightE = ellipse[1][1]
        # Maps rotation to (-90 to 90). Makes it easier to tell direction of slant
        rotation = translateRotation(rotation, widthE, heightE)

        cv2.ellipse(image, ellipse, (23, 184, 80), 3)
        return rotation
    except:
        # Gets rotated bounding rectangle of contour
        rect = cv2.minAreaRect(cnt)
        # Creates box around that rectangle
        box = cv2.boxPoints(rect)
        # Not exactly sure
        box = np.int0(box)
        # Gets center of rotated rectangle
        center = rect[0]
        # Gets rotation of rectangle; same as rotation of contour
        rotation = rect[2]
        # Gets width and height of rotated rectangle
        width = rect[1][0]
        height = rect[1][1]
        # Maps rotation to (-90 to 90). Makes it easier to tell direction of slant
        rotation = translateRotation(rotation, width, height)
        return rotation




team = 2706
server = False
cameraConfigs = []

currentImg = 0

def draw_circle(event,x,y,flags,param):


    if event == cv2.EVENT_LBUTTONDOWN:
        green = np.uint8([[[img[y, x, 0], img[y, x, 1], img[y, x, 2]]]])
        print(x, y, img[y, x, 2], img[y, x, 1], img[y, x, 0], cv2.cvtColor(green,cv2.COLOR_BGR2HSV))


Driver = False
Tape = False
PowerCell = False
ControlPanel = False
OuterTarget = True


img = images[0]

imgLength = len(images)

print("Hello Vision Team!")

while True:

    frame = img

    if Driver:

        processed = frame

    else:

        if Tape:

            threshold = threshold_videho(lower_green, upper_green, frame)
            processed = findTargets(frame, threshold)

        else:
            if PowerCell:
                boxBlur = blurImg(frame, yellow_blur)
                threshold = threshold_video(lower_yellow, upper_yellow, boxBlur)
                processed = findPowerCell(frame, threshold)
            elif ControlPanel:
                boxBlur = blurImg(frame, yellow_blur)
                # cv2.putText(frame, "Find Cargo", (40, 40), cv2.FONT_HERSHEY_COMPLEX, .6, (255, 255, 255))
                threshold = threshold_video(lower_yellow, upper_yellow, frame)
                processed = findControlPanel(frame, threshold)
            
            elif OuterTarget:
                threshold = threshold_video(lower_green, upper_green, frame)
                findTargets(frame, threshold)

    cv2.imshow("raw", img)
    cv2.imshow("threshold", threshold)
    cv2.imshow("processed", processed)
    cv2.setMouseCallback('raw', draw_circle)

    key = cv2.waitKeyEx(0) & 0xFF
    print(key) 

    if key == 27: # <ESC> user wants to exit
        break
    elif key == 112: # 'p' previous image
        if currentImg - 1 < 0:
            currentImg = imgLength - 1
        else:
            currentImg = currentImg - 1
    elif key == 110: # 'n' next image
        if currentImg + 1 > imgLength - 1:
            currentImg = 0
        else:
            currentImg = currentImg + 1

    elif key == ord('h'): # 'h' decrease lower hue
        if (lower_color[0] > 0):
            lower_color[0] = lower_color[0] - 1;
        print()
        print('Lower hue: ', lower_color[0])
    elif key == ord('j'): # 'H' increase lower hue
        if (lower_color[0] < upper_color[0]):
            lower_color[0] = lower_color[0] + 1;
        print()
        print('Lower hue: ', lower_color[0])
    elif key == ord('k'): # 'j' decrease upper hue
        if (upper_color[0] > lower_color[0]):
            upper_color[0] = upper_color[0] - 1;
        print()
        print('Upper hue: ', upper_color[0])
    elif key == ord('l'): # 'J' increase upper hue
        if (upper_color[0] < 255):
            upper_color[0] = upper_color[0] + 1;
        print()
        print('Upper hue: ', upper_color[0])


    elif key == ord('s'): # 's' decrease lower saturation
        if (lower_color[1] > 0):
            lower_color[1] = lower_color[1] - 1;
        print()
        print('Lower saturation: ', lower_color[1])
    elif key == ord('d'): # 'S' increase lower saturation
        if (lower_color[1] < upper_color[1]):
            lower_color[1] = lower_color[1] + 1;
        print()
        print('Lower saturation: ', lower_color[1])
    elif key == ord('f'): # 'd' decrease upper saturation
        if (upper_color[1] > lower_color[1]):
            upper_color[1] = upper_color[1] - 1;
        print()
        print('Upper saturation: ', upper_color[1])
    elif key == ord('g'): # 'D' increase upper saturation
        if (upper_color[1] < 255):
            upper_color[1] = upper_color[1] + 1;
        print()
        print('Upper saturation: ', upper_color[1])


    elif key == ord('x'): # 'v' decrease lower hue value
        if (lower_color[2] > 0):
            lower_color[2] = lower_color[2] - 1;
        print()
        print('Lower value: ', lower_color[2])
    elif key == ord('c'): # 'V' increase lower hue value
        if (lower_color[2] < upper_color[2]):
            lower_color[2] = lower_color[2] + 1;
        print()
        print('Lower value: ', lower_color[2])
    elif key == ord('v'): # 'b' decrease upper hue value
        if (upper_color[2] > lower_color[2]):
            upper_color[2] = upper_color[2] - 1;
        print()
        print('Upper value: ', upper_color[2])
    elif key == ord('b'): # 'B' increase upper hue value
        if (upper_color[2] < 255):
            upper_color[2] = upper_color[2] + 1
        print()
        print('Upper value: ', upper_color[2])

    elif key == 109: # 'm' print hue, saturation, value bounds for mask
        print()
        print('Color bounds for ')
        if color_is_yellow == True:
            print('YELLOW:')
        else:
            print('GREEN:')
        print('Hue:        [', lower_color[0], ',', upper_color[1], ']')
        print('Saturation: [', lower_color[1], ',', upper_color[1], ']')
        print('Value:      [', lower_color[2], ',', upper_color[2], ']')

    elif key == 99: # 'c' toogle between yellow and green
        color_is_yellow = not(color_is_yellow)
        if color_is_yellow == True: # yellow
            lower_color = lower_yellow
            upper_color = upper_yellow
        else: # green
            lower_color = lower_green
            upper_color = upper_green
        print()
        print('color_is_yellow: ', color_is_yellow)


    elif key == 107: # 'k'
        #intMaskMethod = 1
        print()
        print('To be implemented')
    elif key == 109: # 'm'
        #intMaskMethod = 2
        print()
        print('To be implemented')
    elif key == 32: # space
        print()
        print('...repeat...')
    else:
        print ("Unrecognized key: ", key)



    #currentImg += 1
    #print(imgLength)

    #if (currentImg == imgLength-1 ):
    #     currentImg = 0

    img = images[currentImg]





