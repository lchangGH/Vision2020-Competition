#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2018 FIRST. All Rights Reserved.
# Open Source Software - may be modified and shared by FRC teams. The code
# must be accompanied by the FIRST BSD license file in the root directory of
# the project.

# My 2020 license: use it as much as you want. Crediting is recommended because it lets me know that I am being useful.
# Some parts of the architecture are based on 2019 code from the Screaming Chickens 3997

# This is meant to be used in conjuction with WPILib Raspberry Pi image: https://github.com/wpilibsuite/FRCVision-pi-gen
# ----------------------------------------------------------------------------

import json
import time
import sys
from threading import Thread
import random

from cscore import CameraServer, VideoSource
from networktables import NetworkTablesInstance
import cv2
import numpy as np
from networktables import NetworkTables
from networktables.util import ntproperty
import math

########### SET RESOLUTION TO 256x144 !!!! ############

# import the necessary packages
import datetime


# Class to examine Frames per second of camera stream. Currently not used.
class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        if self._end != None:
            return datetime.datetime.now() - self._start
        else:
            return datetime.datetime.now() - self._start

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


# class that runs separate thread for showing video,
class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, imgWidth, imgHeight, cameraServer, frame=None):
        self.outputStream = cameraServer.putVideo("2706_out", imgWidth, imgHeight)
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            self.outputStream.putFrame(self.frame)

    def stop(self):
        self.stopped = True

    def notifyError(self, error):
        self.outputStream.notifyError(error)


# Class that runs a separate thread for reading  camera server also controlling exposure.
class WebcamVideoStream:
    def __init__(self, camera, cameraServer, frameWidth, frameHeight, name="WebcamVideoStream"):
        # initialize the video camera stream and read the first frame
        # from the stream

        # Automatically sets exposure to 0 to track tape
        self.webcam = camera
        self.webcam.setExposureManual(39)
        self.webcam.setExposureAuto()

        # Some booleans so that we don't keep setting exposure over and over to the same value

        self.autoExpose = True
        self.prevValue = True
        # Make a blank image to write on
        self.img = np.zeros(shape=(frameWidth, frameHeight, 3), dtype=np.uint8)
        # Gets the video
        self.stream = cameraServer.getVideo()
        (self.timestamp, self.img) = self.stream.grabFrame(self.img)

        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            global switch
            if self.stopped:
                return

            if switch == 1: #driver mode
                self.autoExpose = True
                #print("Driver mode")
                if self.autoExpose != self.prevValue:
                    self.webcam.setExposureManual(60)
                    self.webcam.setExposureManual(50)
                    self.webcam.setExposureAuto()
                    #print("Driver mode")
                    self.prevValue = self.autoExpose
            elif switch != 1: #not driver mode
                self.autoExpose = False
                #print("Not driver mode")
                if self.autoExpose != self.prevValue:

                    self.webcam.setExposureManual(50)
                    self.webcam.setExposureManual(35)
                    self.webcam.setExposureAuto()
                    #print("Not driver mode")
                    self.prevValue = self.autoExpose

            # gets the image and timestamp from cameraserver
            (self.timestamp, self.img) = self.stream.grabFrame(self.img)

    def read(self):
        # return the frame most recently read
        return self.timestamp, self.img

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def getError(self):
        return self.stream.getError()


###################### PROCESSING OPENCV ################################

# counts frames for writing images
frameStop = 0
ImageCounter = 0

# Angles in radians

# image size ratioed to 4:3
image_width = 640
image_height = 480

# Lifecam 3000 from datasheet
# Datasheet: https://dl2jx7zfbtwvr.cloudfront.net/specsheets/WEBC1010.pdf
diagonalView = math.radians(68.5)

# 16:9 aspect ratio
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
yellow_blur = 3

# define range of green of retroreflective tape in HSV
lower_green = np.array([40, 75, 75])
upper_green = np.array([96, 255, 255])

#lower_yellow = np.array([15, 205, 100])
#upper_yellow = np.array([27, 255, 255])

lower_yellow = np.array([14, 150, 100])
upper_yellow = np.array([30, 255, 255])



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
    combined_mask = cv2.bitwise_and(h, cv2.bitwise_and(s,v))
    
    # hold the HSV image to get only red colors
    #mask = cv2.inRange(combined, lower_color, upper_color)

    # Returns the masked imageBlurs video to smooth out image
    global frameStop
    if frameStop == 1:
        global ImageCounter, matchNumber, matchNumberDefault
        matchNumber = networkTableMatch.getNumber("MatchNumber", 0)
        if matchNumber == 0:
            matchNumber = matchNumberDefault
        cv2.imwrite('/mnt/VisionImages/visionImg-' + str(matchNumber) + "-" + str(ImageCounter) + '_mask.png',
                    combined_mask)
    return combined_mask


# Finds the tape targets from the masked image and displays them on original stream + network tales
def findTargets(frame, mask):

    global networkTable
    if networkTable.getBoolean("SendMask", False):
        return mask

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
        image = findTape(contours, image, centerX, centerY)
    # Shows the contours overlayed on the original video
    return image


# Finds the balls from the masked image and displays them on original stream + network tables
def findPowerCell(frame, mask):
    
    global networkTable
    if networkTable.getBoolean("SendMask", False):
        return mask

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


# Draws Contours and finds center and yaw of orange ball
# centerX is center x coordinate of image
# centerY is center y coordinate of image
def findBall(contours, image, centerX, centerY):
    screenHeight, screenWidth, channels = image.shape
    # Seen vision targets (correct angle, adjacent to each other)

    if len(contours) > 0:
        # Sort contours by area size (biggest to smallest)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        cntHeight = 0
        biggestPowerCell = []
        for cnt in cntsSorted:
            x, y, w, h = cv2.boundingRect(cnt)

            #print("bounding rec x: " + str(y))
            #print("bounding rec y: " + str(x))
            #print("bounding rec height: " + str(h))
            #print("bounding rec width: " + str(w))
        
            cntHeight = h
            aspect_ratio = float(w) / h
            # Get moments of contour; mainly for centroid
            M = cv2.moments(cnt)
            # Get convex hull (bounding polygon on contour)
            #hull = cv2.convexHull(cnt)
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
                    # Covert boxpoints to integer
                    box = np.int0(box)
                    # Draws rotated rectangle
                    #cv2.drawContours(image, [box], 0, (23, 184, 80), 3)

                    # Draws a vertical white line passing through center of contour
                    cv2.line(image, (cx, screenHeight), (cx, 0), (255, 255, 255))
                    # Draws a white circle at center of contour
                    cv2.circle(image, (cx, cy), 6, (255, 255, 255))

                    # Draws the contours
                    cv2.drawContours(image, [cnt], 0, (23, 184, 80), 1)

                    # Gets the (x, y) and radius of the enclosing circle of contour
                    #(x, y), radius = cv2.minEnclosingCircle(cnt)
                    # Rounds center of enclosing circle
                    #center = (int(x), int(y))
                    # Rounds radius of enclosning circle
                    #radius = int(radius)
                    # Makes bounding rectangle of contour
                    #rx, ry, rw, rh = cv2.boundingRect(cnt)
                    #x, y, w, h = cv2.boundingRect(cnt)

                    # Draws contour of bounding rectangle in red
                    #cv2.rectangle(image, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 1)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                   
                    # Draws circle in cyan
                    #cv2.circle(image, center, radius, (255, 255,0), 1)

                    # Appends important info to array
                    if [cx, cy, cnt, cntHeight] not in biggestPowerCell:
                        biggestPowerCell.append([cx, cy, cnt, cntHeight, aspect_ratio])

        # Check if there are PowerCell seen
        if (len(biggestPowerCell) > 0):
            # pushes that it sees cargo to network tables

            finalTarget = []
            # Sorts targets based on largest height
            biggestPowerCell.sort(key=lambda height: math.fabs(height[3]))

            #sorts closestPowerCell - contains center-x, center-y, contour and contour height from the
            #bounding rectangle.  The closest one has the largest height
            closestPowerCell = min(biggestPowerCell, key=lambda height: (math.fabs(height[3] - centerX)))

            # extreme points
            leftmost = tuple(closestPowerCell[2][closestPowerCell[2][:,:,0].argmin()][0])
            rightmost = tuple(closestPowerCell[2][closestPowerCell[2][:,:,0].argmax()][0])
            topmost = tuple(closestPowerCell[2][closestPowerCell[2][:,:,1].argmin()][0])
            bottommost = tuple(closestPowerCell[2][closestPowerCell[2][:,:,1].argmax()][0])

            # draw extreme points
            # from https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
            cv2.circle(image, leftmost, 8, (0,255,0), -1)
            cv2.circle(image, rightmost, 8, (0,0,255), -1)
            cv2.circle(image, topmost, 8, (255,255,255), -1)
            cv2.circle(image, bottommost, 8, (255,0,0), -1)
            #print('extreme points', leftmost,rightmost,topmost,bottommost)

            #print("topmost: " + str(topmost[0]))
            #print("bottommost: " + str(bottommost[0]))
            #xCoord of the closest ball will be the x position differences between the topmost and 
            #bottom most points
            if (topmost[0] > bottommost[0]):
                xCoord = int(round((topmost[0]-bottommost[0])/2)+bottommost[0])
            else: 
                xCoord = int(round((bottommost[0]-topmost[0])/2)+topmost[0])

            print(xCoord)
            #If aspect ratio is greater than 0.8 and less than 1.2, treat it as a single ball
            #and simply use the center x value (cx)
            if (closestPowerCell[4] > 0.9 and closestPowerCell[4] < 1.2):
                xCoord = closestPowerCell[0]

            finalTarget.append(calculateYaw(xCoord, centerX, H_FOCAL_LENGTH))
            finalTarget.append(calculateDistWPILib(closestPowerCell[3]))

            # Puts the yaw on screen
            # Draws yaw of target + line where center of target is
            finalYaw = round(finalTarget[1]*1000)/1000
            cv2.putText(image, "Yaw: " + str(finalTarget[0]), (40, 40), cv2.FONT_HERSHEY_COMPLEX, .6,
                        (255, 255, 255))
            cv2.putText(image, "Dist: " + str(finalYaw), (40, 100), cv2.FONT_HERSHEY_COMPLEX, .6,
                        (255, 255, 255))
            cv2.line(image, (xCoord, screenHeight), (xCoord, 0), (255, 0, 0), 2)

            currentAngleError = finalTarget[0]
            # pushes powerCell angle to network tables
            networkTable.putNumber("YawToPowerCell", finalTarget[0])
            networkTable.putNumber("DistanceToPowerCell", finalYaw)

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




            #print("The contour height is, ", cntHeight)
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
        global fps
        networkTable.putBoolean("tapeDetected", True)
        # Sorts targets based on x coords to break any angle tie
        targets.sort(key=lambda x: math.fabs(x[0]))
        finalTarget = min(targets, key=lambda x: math.fabs(x[1]))
        # Puts the yaw on screen
        # Draws yaw of target + line where center of target is
        cv2.putText(image, "Yaw: " + str(finalTarget[1]), (40, 40), cv2.FONT_HERSHEY_COMPLEX, .5,
                    (255, 255, 255))
        cv2.putText(image, "Dist: " + str(finalTarget[2]), (40, 90), cv2.FONT_HERSHEY_COMPLEX, .5,
                    (255, 255, 255))
        cv2.putText(image, "Time: " + str(fps.elapsed()), (40, 140), cv2.FONT_HERSHEY_COMPLEX, .5,
                    (255, 255, 255))
        cv2.line(image, (finalTarget[0], screenHeight), (finalTarget[0], 0), (255, 0, 0), 2)

        currentAngleError = finalTarget[1]
        # pushes vision target angle to network tables
        networkTable.putNumber("tapeYaw", currentAngleError)
        if finalTarget[2] < 6 and finalTarget[2] > 3:
            if currentAngleError > -(10 - finalTarget[2]) and currentAngleError < (10 - finalTarget[2]):

                networkTable.putBoolean("Aligned", True)

            else:

                networkTable.putBoolean("Aligned", False)
        else:
            networkTable.putBoolean("Aligned", False)


        # pushes distance to network table
        networkTable.putNumber("distance", finalTarget[2])

        vectorCameraToTarget = ntproperty('/PathFinder/vectorCameraToTarget',[currentAngleError,finalTarget[2]])
        # networkTable.putNumber("vectorCameraToTarget",[currentAngleError,finalTarget[2]])
    else:
        # pushes that it deosn't see vision target to network tables
        networkTable.putBoolean("tapeDetected", False)

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
    
    return cntSize > (image_width / 7)


# Checks if ball contours are worthy based off of contour area and (not currently) hull area
def checkBall(cntSize, cntAspectRatio):
    return (cntSize > (image_width / 2)) and (round(cntAspectRatio) == 1)

def checkHatch(cntSize, cntAspectRatio):
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

avg = [0 for i in range(0, 4)]

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


#################### FRC VISION PI Image Specific #############
configFile = "/boot/frc.json"


class CameraConfig: pass


team = 2706
server = False
cameraConfigs = []

"""Report parse error."""


def parseError(str):
    print("config error in '" + configFile + "': " + str, file=sys.stderr)


"""Read single camera configuration."""


def readCameraConfig(config):
    cam = CameraConfig()

    # name
    try:
        cam.name = config["name"]
    except KeyError:
        parseError("could not read camera name")
        return False

    # path
    try:
        cam.path = config["path"]
    except KeyError:
        parseError("camera '{}': could not read path".format(cam.name))
        return False

    cam.config = config

    cameraConfigs.append(cam)
    return True


"""Read configuration file."""


def readConfig():
    global team
    global server

    # parse file
    try:
        with open(configFile, "rt") as f:
            j = json.load(f)
    except OSError as err:
        print("could not open '{}': {}".format(configFile, err), file=sys.stderr)
        return False

    # top level must be an object
    if not isinstance(j, dict):
        parseError("must be JSON object")
        return False

    # team number
    try:
        team = j["team"]
    except KeyError:
        parseError("could not read team number")
        return False

    # ntmode (optional)
    if "ntmode" in j:
        str = j["ntmode"]
        if str.lower() == "client":
            server = False
        elif str.lower() == "server":
            server = True
        else:
            parseError("could not understand ntmode value '{}'".format(str))

    # cameras
    try:
        cameras = j["cameras"]
    except KeyError:
        parseError("could not read cameras")
        return False
    for camera in cameras:
        if not readCameraConfig(camera):
            return False

    return True


"""Start running the camera."""


def startCamera(config):
    print("Starting camera '{}' on {}".format(config.name, config.path))
    cs = CameraServer.getInstance()
    camera = cs.startAutomaticCapture(name=config.name, path=config.path)

    camera.setConfigJson(json.dumps(config.config))

    return cs, camera

start, switched, prevCam = True, False, 0

currentCam = 0

def switchCam():
    global currentCam, webcam, cameras, streams, cameraServer, cap, image_width, image_height, prevCam
    if networkTable.getNumber("Cam", 1):
        currentCam = 1
    else:
        currentCam = 0
    prevCam = currentCam
    cap.stop()
    webcam = cameras[currentCam]
    cameraServer = streams[currentCam]
    # Start thread reading camera
    cap = WebcamVideoStream(webcam, cameraServer, image_width, image_height).start()


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        configFile = sys.argv[1]
    # read configuration
    if not readConfig():
        sys.exit(1)

    # start NetworkTables
    ntinst = NetworkTablesInstance.getDefault()
    # Name of network table - this is how it communicates with robot. IMPORTANT
    networkTable = NetworkTables.getTable('MergeVision')


    networkTableMatch = NetworkTables.getTable("FMSInfo")

    networkTableTime = NetworkTables.getTable("SmartDashboard")

    if server:
        print("Setting up NetworkTables server")
        ntinst.startServer()
    else:
        print("Setting up NetworkTables client for team {}".format(team))
        ntinst.startClientTeam(team)

    # start cameras
    cameras = []
    streams = []
    for cameraConfig in cameraConfigs:
        cs, cameraCapture = startCamera(cameraConfig)
        streams.append(cs)
        cameras.append(cameraCapture)
    # Get the first camera

    webcam = cameras[currentCam]
    cameraServer = streams[currentCam]
    # Start thread reading camera
    cap = WebcamVideoStream(webcam, cameraServer, image_width, image_height).start()
    # cap = cap.findTape
    # (optional) Setup a CvSource. This will send images back to the Dashboard
    # Allocating new images is very expensive, always try to preallocate
    img = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8)
    # Start thread outputing stream
    streamViewer = VideoShow(image_width, image_height, cameraServer, frame=img).start()

    # cap.autoExpose=True;
    tape = True
    fps = FPS().start()
    # TOTAL_FRAMES = 200;
    # loop forever
    networkTable.putBoolean("Driver", False)
    networkTable.putBoolean("Tape", False)
    networkTable.putBoolean("PowerCell", True)
    networkTable.putBoolean("ControlPanel", False)
    networkTable.putBoolean("WriteImages", False)
    networkTable.putBoolean("SendMask", False)
    networkTable.putBoolean("TopCamera", False)
    networkTable.putBoolean("Cam", currentCam)
    networkTable.putBoolean("Aligned", False)

    matchNumberDefault = random.randint(1, 1000)



    processed = 0

    while True:

        if networkTableTime.getNumber("Match Time", 1) == 0:
            networkTable.putBoolean("WriteImages", False)

        if networkTable.getBoolean("TopCamera", False):
            currentCam = 1
        else:
            currentCam = 0

        if networkTable.getNumber("Cam", currentCam) != prevCam:
            switchCam()

        # Tell the CvSink to grab a frame from the camera and put it
        # in the source image.  If there is an error notify the output.
        timestamp, img = cap.read()
        if frameStop == 0:
            matchNumber = networkTableMatch.getNumber("MatchNumber", 0)
            if matchNumber == 0:
                matchNumber = matchNumberDefault
            cv2.imwrite('/mnt/VisionImages/visionImg-' + str(matchNumber) + "-" + str(ImageCounter) + '_Raw.png',
                        img)
        # Uncomment if camera is mounted upside down
        if networkTable.getBoolean("TopCamera", False):
            frame = flipImage(img)
        else:
            frame = img
        # Comment out if camera is mounted upside down
        # img = findCargo(frame,img)



        
        if timestamp == 0:
            # Send the output the error.
            streamViewer.notifyError(cap.getError())
            # skip the rest of the current iteration
            continue
        # Checks if you just want camera for driver (No processing), False by default




        switch = 0

        if (networkTable.getBoolean("Tape", True)):
            if switch != 2:
                print("finding tape")
            switch = 2
            # Lowers exposure to 0
            #cap.autoExpose = False
            #cap.webcam.setExposureManual(50)
            #cap.webcam.setExposureManual(20)
            #boxBlur = blurImg(frame, green_blur)
            # cv2.putText(frame, "Find Tape", (40, 40), cv2.FONT_HERSHEY_COMPLEX, .6, (255, 255, 255))
            threshold = threshold_video(lower_green, upper_green, frame)
            processed = findTargets(frame, threshold)

        else:
            if (networkTable.getBoolean("PowerCell", True)):
                # Checks if you just want camera for Cargo processing, by dent of everything else being false, true by default
                #if (networkTable.getBoolean("Cargo", True)):
                if switch != 3:
                    print("find Power Cell")
                switch = 3
                #cap.autoExpose = True
                boxBlur = blurImg(frame, yellow_blur)
                # cv2.putText(frame, "Find Cargo", (40, 40), cv2.FONT_HERSHEY_COMPLEX, .6, (255, 255, 255))
                threshold = threshold_video(lower_yellow, upper_yellow, boxBlur)
                processed = findPowerCell(frame, threshold)
            elif (networkTable.getBoolean("ControlPanel", True)):
                # Checks if you just want camera for Control Panel, by dent of everything else being false, true by default
                #if (networkTable.getBoolean("Cargo", True)):
                if switch != 4:
                    print("find Control Panel Colour")
                switch = 4
                #cap.autoExpose = True
                boxBlur = blurImg(frame, yellow_blur)
                # Need to create proper mask for control panel
                threshold = threshold_video(lower_yellow, upper_yellow, boxBlur)
                processed = findControlPanel(frame, threshold)

        # Puts timestamp of camera on netowrk tables
        networkTable.putNumber("VideoTimestamp", timestamp)



        if (networkTable.getBoolean("WriteImages", True)):
            frameStop = frameStop + 1
            if frameStop == 15 :
                matchNumber = networkTableMatch.getNumber("MatchNumber", 0)
                if matchNumber == 0:
                    matchNumber = matchNumberDefault
                cv2.imwrite('/mnt/VisionImages/visionImg-' +str(matchNumber)+"-"+ str(ImageCounter) + '_Proc.png', processed)
                frameStop = 0
                ImageCounter = ImageCounter+1
                if (ImageCounter==10000):
                    ImageCounter=0

        # networkTable.putBoolean("Driver", True)
        streamViewer.frame = processed
        # update the FPS counter
        fps.update()
        # Flushes camera values to reduce latency
        ntinst.flush()
    # Doesn't do anything at the moment. You can easily get this working by indenting these three lines
    # and setting while loop to: while fps._numFrames < TOTAL_FRAMES
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
