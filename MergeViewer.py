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
images = load_images_from_folder("./OuterTargetHalfScale")
#images = load_images_from_folder("./OuterTargetSketchup")
#images = load_images_from_folder("./PowerCell25Scale")
#images = load_images_from_folder("./PowerCellImages")
#images = load_images_from_folder("./PowerCellFullScale")
#images = load_images_from_folder("./PowerCellFullMystery")
#images = load_images_from_folder("./PowerCellSketchup")
#images = load_images_from_folder("./PowerCellSketchup")
#images = load_images_from_folder("./LifeCamPhotos")


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
yellow_blur = 3

# define range of green of retroreflective tape in HSV
lower_green = np.array([23, 50, 35])
upper_green = np.array([85, 255, 255])

lower_yellow = np.array([20, 35, 100])
upper_yellow = np.array([60, 255, 255])

switch = 1

# real world dimensions of the goal target
# These are the full dimensions around both strips
TARGET_STRIP_LENGTH = 19.625    # inches
TARGET_HEIGHT = 17.0            # inches@!
TARGET_TOP_WIDTH = 39.25        # inches
TARGET_BOTTOM_WIDTH = TARGET_TOP_WIDTH - 2*TARGET_STRIP_LENGTH*math.cos(math.radians(60))

#This is the X position difference between the upper target length and corner point
TARGET_BOTTOM_CORNER_WIDTH = TARGET_STRIP_LENGTH*math.cos(math.radians(60))

# [0, 0] is center of the quadrilateral drawn around the high goal target
# [top_left, bottom_left, bottom_right, top_right]
# real_world_coordinates = np.array([
#     [-TARGET_TOP_WIDTH / 2, TARGET_HEIGHT / 2, 0.0],
#     [-TARGET_BOTTOM_WIDTH / 2, -TARGET_HEIGHT / 2, 0.0],
#     [TARGET_BOTTOM_WIDTH / 2, -TARGET_HEIGHT / 2, 0.0],
#     [TARGET_TOP_WIDTH / 2, TARGET_HEIGHT / 2, 0.0]
# ])


# real_world_coordinates = np.array([
#     [-TARGET_TOP_WIDTH / 2, TARGET_HEIGHT / 2, 0.0],
#     [-TARGET_BOTTOM_WIDTH / 2, -TARGET_HEIGHT / 2, 0.0],
#     [TARGET_BOTTOM_WIDTH / 2, -TARGET_HEIGHT / 2, 0.0],
#     [TARGET_TOP_WIDTH / 2, TARGET_HEIGHT / 2, 0.0]
# ])

#top_left, top_right, bottom_left, bottom_right
real_world_coordinates = np.array([
        [0.0, 0.0, 0.0],             # Top Left point
        [TARGET_TOP_WIDTH, 0.0, 0.0],           # Top Right Point
        [TARGET_BOTTOM_CORNER_WIDTH, TARGET_HEIGHT, 0.0],            # Bottom Left point
        [TARGET_TOP_WIDTH-TARGET_BOTTOM_CORNER_WIDTH, TARGET_HEIGHT, 0.0]          # Bottom Right point
    ])



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
    
    #show the mask
    cv2.imshow("mask", combined_mask)

    # hold the HSV image to get only red colors
    # mask = cv2.inRange(combined, lower_color, upper_color)

    # Returns the masked imageBlurs video to smooth out image

    return combined_mask


# Finds the tape targets from the masked image and displays them on original stream + network tales
def findTargets(frame, mask):


    # Finds contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

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
        image = findBall(contours, image, centerX, centerY)
    # Shows the contours overlayed on the original video
    return image



# Draws Contours and finds center and yaw of orange ball
# centerX is center x coordinate of image
# centerY is center y coordinate of image
def findBall(contours, image, centerX, centerY):
    screenHeight, screenWidth, channels = image.shape
    # Seen vision targets (correct angle, adjacent to each other)
    #cargo = []

    if len(contours) > 0:
        # Sort contours by area size (biggest to smallest)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        cntHeight = 0
        biggestPowerCell = []
        for cnt in cntsSorted:
            x, y, w, h = cv2.boundingRect(cnt)

            #print("bounding rec height: " + str(h))
            #print("bounding rec width: " + str(w))
            print("bounding rec x: " + str(y))
            print("bounding rec y: " + str(x))
            print("bounding rec height: " + str(h))
            print("bounding rec width: " + str(w))
        
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
                        biggestPowerCell.append([cx, cy, cnt, cntHeight])

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
            cv2.circle(image, leftmost, 12, (0,255,0), -1)
            cv2.circle(image, rightmost, 12, (0,0,255), -1)
            cv2.circle(image, topmost, 12, (255,255,255), -1)
            cv2.circle(image, bottommost, 12, (255,0,0), -1)
            print('extreme points', leftmost,rightmost,topmost,bottommost)

            print("topmost: " + str(topmost[0]))
            print("bottommost: " + str(bottommost[0]))
            #xCoord of the closest ball will be the x position differences between the topmost and 
            #bottom most points
            if (topmost[0] > bottommost[0]):
                xCoord = int(round((topmost[0]-bottommost[0])/2)+bottommost[0])
            else: 
                xCoord = int(round((bottommost[0]-topmost[0])/2)+topmost[0])

            print(xCoord)
            if (aspect_ratio > 0.9 and aspect_ratio < 1.1):
                xCoord = closestPowerCell[0]

            finalTarget.append(calculateYaw(xCoord, centerX, H_FOCAL_LENGTH))
            finalTarget.append(calculateDistWPILib(closestPowerCell[3]))
            #print("Yaw: " + str(finalTarget[0]))

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


def order_points(pts):
	# initialize a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype="float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[3] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[2] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
    #global warped
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped    

""" # # 3D Rotation estimation
# def findTvecRvec(image, topLeft, topRight, bottomLeft, bottomRight):
#     # Read Image
#     #im = cv2.imread("headPose.jpg");
#     size = image.shape
     
#     #2D image points. If you change the image, you need to change vector
#     image_points = np.array([
#                             topLeft,     # Top Left point
#                             topRight,    # Top Right point
#                             bottomLeft,  # Bottom Left
#                             bottomRight, # Bottom right point
#                         ], dtype="double")
 
#     # 3D model points.
#     model_points = np.array([
#                             (0.0, 0.0, 0.0),             # Top Left point
#                             (39.25, 0.0, 0.0),           # Top Right Point
#                             (0.0, 17.0, 0.0),            # Bottom Left point
#                             (39.25, 17.0, 0.0),          # Bottom Right point
#                         ])
 
 
#     # Camera internals
 
#     focal_length = size[1]
#     center = (size[1]/2, size[0]/2)
#     camera_matrix = np.array(
#                          [[H_FOCAL_LENGTH, 0, center[0]],
#                          [0, V_FOCAL_LENGTH, center[1]],
#                          [0, 0, 1]], dtype = "double"
#                          )
 
#     print("Camera Matrix :\n {0}".format(camera_matrix))
 
#     dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
#     (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
 
#     print ("Rotation Vector:\n {0}".format(rotation_vector))
#     print ("Translation Vector:\n {0}".format(translation_vector))
#     return rotation_vector, translation_vector """


# 3D Rotation estimation
def findTvecRvec(image, outer_corners):
    # Read Image
    size = image.shape
 
    # Camera internals
 
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                         [[H_FOCAL_LENGTH, 0, center[0]],
                         [0, V_FOCAL_LENGTH, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
 
    print("Camera Matrix :\n {0}".format(camera_matrix))
 
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(real_world_coordinates, outer_corners, camera_matrix, dist_coeffs)
 
    print ("Rotation Vector:\n {0}".format(rotation_vector))
    print ("Translation Vector:\n {0}".format(translation_vector))
    return success, rotation_vector, translation_vector


def contour_center_width(contour):
    '''Find boundingRect of contour, but return center and width/height'''

    x, y, w, h = cv2.boundingRect(contour)
    return (x + int(w / 2), y + int(h / 2)), (w, h)


# def test_candidate_contour(self, candidate, shape):
#         '''Determine the true target contour out of potential candidates'''

#         cand_width = candidate['widths'][0]
#         cand_height = candidate['widths'][1]

#         cand_dim_ratio = cand_width / cand_height
#         if cand_dim_ratio < self.min_dim_ratio:
#             return None
#         cand_area_ratio = cv2.contourArea(candidate["contour"]) / (cand_width * cand_height)
#         if cand_area_ratio > self.max_area_ratio:
#             return None

#         hull = cv2.convexHull(candidate['contour'])
#         contour = quad_fit(hull)

#         if contour is not None and len(contour) == 4:
#             return contour

#         return None

#@staticmethod
def sort_corners(contour, center=None):
    '''Sort the contour in our standard order, starting upper-left and going counter-clockwise'''

    # Note: the inputs are all numpy arrays, so it is fast to operate on the whole array at once

    if center is None:
        center = contour.mean(axis=0)
        d = contour - center
        # remember that y-axis increases down, so flip the sign
        angle = (np.arctan2(-d[:, 1], d[:, 0]) - pi_by_2) % two_pi
        return contour[np.argsort(angle)]

#Computer the final output values, 
#angle 1 is the Yaw to the target
#distance is the distance to the target
#angle 2 is the Yaw of the Robot to the target
def compute_output_values(rvec, tvec):
    '''Compute the necessary output distance and angles'''

    # The tilt angle only affects the distance and angle1 calcs
    # This is a major impact on calculations
    tilt_angle = math.radians(20)

    x = tvec[0][0]
    z = math.sin(tilt_angle) * tvec[1][0] + math.cos(tilt_angle) * tvec[2][0]

    # distance in the horizontal plane between camera and target
    distance = math.sqrt(x**2 + z**2)

    # horizontal angle between camera center line and target
    angle1 = math.atan2(x, z)

    rot, _ = cv2.Rodrigues(rvec)
    rot_inv = rot.transpose()
    pzero_world = np.matmul(rot_inv, -tvec)
    angle2 = math.atan2(pzero_world[0][0], pzero_world[2][0])

    return distance, angle1, angle2



# Draws Contours and finds center and yaw of vision targets
# centerX is center x coordinate of image
# centerY is center y coordinate of image
def findTape(contours, image, centerX, centerY):

    #global warped
    screenHeight, screenWidth, channels = image.shape
    # Seen vision targets (correct angle, adjacent to each other)
    targets = []

    if len(contours) >= 2:
        # Sort contours by area size (biggest to smallest)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[:1]

       
        for cnt in cntsSorted:
            x, y, w, h = cv2.boundingRect(cnt)

            # extreme points
            leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
            rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
            topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
            bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

            # draw extreme points
            # from https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
            cv2.circle(image, leftmost, 6, (0,255,0), -1)
            cv2.circle(image, rightmost, 6, (0,0,255), -1)
            cv2.circle(image, topmost, 6, (255,255,255), -1)
            cv2.circle(image, bottommost, 6, (255,0,0), -1)
            #print('extreme points', leftmost,rightmost,topmost,bottommost)

            #outer_corners = [leftmost, rightmost, topmost, bottommost]

            outer_corners = np.array([leftmost, rightmost, bottommost], dtype="double")
            sorted_corners = order_points(outer_corners)
   
            success, rvec, tvec = findTvecRvec(image, sorted_corners) 

            #retval, rvec, tvec = cv2.solvePnP(self.real_world_coordinates, outer_corners,
            #                                  self.cameraMatrix, self.distortionMatrix)
            if success:
                #result = [1.0, self.finder_id, ]
                #result.extend(compute_output_values(rvec, tvec))
                #return result
                distance, angle1, angle2 = compute_output_values(rvec, tvec)
                cv2.putText(image, "TargetYawToCenter: " + str(angle1), (40, 140), cv2.FONT_HERSHEY_COMPLEX, .6,(255, 255, 255))
                cv2.putText(image, "Distance: " + str(distance/12), (40, 190), cv2.FONT_HERSHEY_COMPLEX, .6,(255, 255, 255))
                cv2.putText(image, "RobotYawToTarget: " + str(angle2), (40, 240), cv2.FONT_HERSHEY_COMPLEX, .6,(255, 255, 255))
                #cv2.line(image, (finalTarget[0], screenHeight), (finalTarget[0], 0), (255, 0, 0), 2)

    #     currentAngleError = finalTarget[1]
    #     # pushes vision target angle to network table


    # cv2.line(image, (round(centerX), screenHeight), (round(centerX), 0), (255, 255, 255), 2)



        #biggestCnts = []
        #for cnt in cntsSorted:
         #   print("Contour: " + str(len(cnt)))
          #  print("List" + str(list(cnt)))
            # Get moments of contour; mainly for centroid
           # M = cv2.moments(cnt)
            # Get convex hull (bounding polygon on contour)
            #hull = cv2.convexHull(cnt)
            # Calculate Contour area
            #cntArea = cv2.contourArea(cnt)
            # calculate area of convex hull
            #hullArea = cv2.contourArea(hull)

            #x, y, w, cntHeight = cv2.boundingRect(cnt)
            #print("contour height of boundingRect: " + str(cntHeight))

             # extreme points
            #leftmost = tuple(cnt[2][cnt[2][:,:,0].argmin()][0])
            #rightmost = tuple(cnt[2][cnt[2][:,:,0].argmax()][0])
            #topmost = tuple(closestPowerCell[2][closestPowerCell[2][:,:,1].argmin()][0])
            #bottommost = tuple(closestPowerCell[2][closestPowerCell[2][:,:,1].argmax()][0])

            # draw extreme points
            # from https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
            #cv2.circle(image, leftmost, 6, (0,255,0), -1)
            #cv2.circle(image, rightmost, 6, (0,0,255), -1)
            #cv2.circle(image, topmost, 6, (255,255,255), -1)
            #cv2.circle(image, bottommost, 6, (255,0,0), -1)

            #pts, dim, a = cv2.minAreaRect(cnt)

            #rect = cv2.minAreaRect(cnt)
            # Creates box around that rectangle
            #box = cv2.boxPoints(rect)
            # Unpack points into integers
            #box = np.int0(box)

            # Copies frame and stores it in image
            #copyimage = image.copy()
            #transform image
            #warped = four_point_transform(copyimage, box)
            #rvec, tvec = findTvecRvec(image, box[0], box[1], box[2], box[3])
            #print("warped: " + str(warped))
            #print("rvec: " + str(rvec))
            #print("tvec: " + str(tvec))

            #cv2.imshow("Warped", warped)
            #cv2.waitKey(0)

            # crop
            #img_croped = crop_minAreaRect(image, rect)
            #print("image cropped:" + str(img_croped))
            #cv2.drawContours(image, img_croped, 0, (0, 255, 255), 6)


            #warped = crop_rotated_rectangle(image, pts)
            #rect = cv2.boundingRect(cnt)
            #warped = crop_minAreaRect(image, rect)

            # print("dim: " + str(dim))
            # print("dim 1: " + str(dim[1]))

            # x = pts[0]
            # y = pts[1]

            # if dim[0] > dim[1]:
            #     cntHeight = dim[0]
            # else:
            #     cntHeight = dim[1]

            # # print("The contour height is, ", cntHeight)
            # # Filters contours based off of size
            # if (checkContours(cntArea, hullArea)):
            #     ### MOSTLY DRAWING CODE, BUT CALCULATES IMPORTANT INFO ###
            #     # Gets the centeroids of contour
            #     if M["m00"] != 0:
            #         cx = int(M["m10"] / M["m00"])
            #         cy = int(M["m01"] / M["m00"])
            #         theCX = cx
            #         theCY = cy
            #     else:
            #         cx, cy = 0, 0
            #     if (len(biggestCnts) < 5):
            #         #### CALCULATES ROTATION OF CONTOUR BY FITTING ELLIPSE ##########
            #         rotation = getEllipseRotation(image, cnt)

            #         # Calculates yaw of contour (horizontal position in degrees)
            #         yaw = calculateYaw(cx, centerX, H_FOCAL_LENGTH)
            #         # Calculates yaw of contour (horizontal position in degrees)
            #         pitch = calculatePitch(cy, centerY, V_FOCAL_LENGTH)
            #         # Calculates Distance
            #         dist = calculateDistance(1, 2, pitch);

            #         ##### DRAWS CONTOUR######
            #         # Gets rotated bounding rectangle of contour
            #         rect = cv2.minAreaRect(cnt)
            #         # Creates box around that rectangle
            #         box = cv2.boxPoints(rect)
            #         # Not exactly sure
            #         box = np.int0(box)
            #         # Draws rotated rectangle in YELLOW
            #         cv2.drawContours(image, [box], 0, (0, 255, 255), 3)

            #         slope = (((box[2,1]-box[1,1])*-1)/(box[2,0]-box[1,0]))
            #         print("box points: " + str(box))
            #         print("slope1: " + str(slope))

            #         # Calculates yaw of contour (horizontal position in degrees)
            #         #yaw = calculateYaw(cx, centerX, H_FOCAL_LENGTH)
            #         # Calculates yaw of contour (horizontal position in degrees)
            #         #pitch = calculatePitch(cy, centerY, V_FOCAL_LENGTH)
            #         # Calculates Distance
            #         #dist = calculateDistance(1, 2, pitch);

    #                 # Draws a vertical white line passing through center of contour
    #                 cv2.line(image, (cx, screenHeight), (cx, 0), (255, 255, 255))
    #                 # Draws a white circle at center of contour
    #                 cv2.circle(image, (cx, cy), 6, (255, 255, 255))

    #                 # Draws the contours
    #                 cv2.drawContours(image, [cnt], 0, (23, 184, 80), 1)

    #                 # Gets the (x, y) and radius of the enclosing circle of contour
    #                 (x, y), radius = cv2.minEnclosingCircle(cnt)
    #                 # Rounds center of enclosing circle
    #                 center = (int(x), int(y))
    #                 # Rounds radius of enclosning circle
    #                 radius = int(radius)
    #                 # Makes bounding rectangle of contour
    #                 rx, ry, rw, rh = cv2.boundingRect(cnt)
    #                 boundingRect = cv2.boundingRect(cnt)
    #                 # Draws countour of bounding rectangle and enclosing circle in blue
    #                 cv2.rectangle(image, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 1)

    #                 cv2.circle(image, center, radius, (255, 0, 0), 1)

    #                 # Appends important info to array
    #                 if [cx, cy, rotation, cnt, cntHeight] not in biggestCnts:
    #                     biggestCnts.append([cx, cy, rotation, cnt, cntHeight])

    #     # Sorts array based on coordinates (leftmost to rightmost) to make sure contours are adjacent
    #     biggestCnts = sorted(biggestCnts, key=lambda x: x[0])
    #     # Target Checking
    #     for i in range(len(biggestCnts) - 1):
    #         # Rotation of two adjacent contours
    #         tilt1 = biggestCnts[i][2]
    #         tilt2 = biggestCnts[i + 1][2]

    #         # x coords of contours
    #         cx1 = biggestCnts[i][0]
    #         cx2 = biggestCnts[i + 1][0]

    #         cy1 = biggestCnts[i][1]
    #         cy2 = biggestCnts[i + 1][1]
    #         # If contour angles are opposite
    #         if (np.sign(tilt1) != np.sign(tilt2)):
    #             centerOfTarget = math.floor((cx1 + cx2) / 2)
    #             # ellipse negative tilt means rotated to right
    #             # Note: if using rotated rect (min area rectangle)
    #             #      negative tilt means rotated to left
    #             # If left contour rotation is tilted to the left then skip iteration
    #             if (tilt1 > 0):
    #                 if (cx1 < cx2):
    #                     continue
    #             # If left contour rotation is tilted to the left then skip iteration
    #             if (tilt2 > 0):
    #                 if (cx2 < cx1):
    #                     continue
    #             # Angle from center of camera to target (what you should pass into gyro)
    #             yawToTarget = calculateYaw(centerOfTarget, centerX, H_FOCAL_LENGTH)
    #             pitchToTarget = calculatePitch(theCY, centerY, H_FOCAL_LENGTH)
    #             # distToTarget = calculateDistance(1, 2, pitchToTarget)
    #             distToTarget = calculateDistWPILib(biggestCnts[i][4])
    #             # Make sure no duplicates, then append
    #             if [centerOfTarget, yawToTarget, distToTarget] not in targets:
    #                 targets.append([centerOfTarget, yawToTarget, distToTarget])
    # # Check if there are targets seen
    # if (len(targets) > 0):
    #     # pushes that it sees vision target to network tables

    #     # Sorts targets based on x coords to break any angle tie
    #     targets.sort(key=lambda x: math.fabs(x[0]))
    #     finalTarget = min(targets, key=lambda x: math.fabs(x[1]))
    #     # Puts the yaw on screen
    #     # Draws yaw of target + line where center of target is
    #     cv2.putText(image, "Yaw: " + str(finalTarget[1]), (40, 40), cv2.FONT_HERSHEY_COMPLEX, .6,
    #                 (255, 255, 255))
    #     cv2.putText(image, "Dist: " + str(finalTarget[2]), (40, 90), cv2.FONT_HERSHEY_COMPLEX, .6,
    #                 (255, 255, 255))
    #     cv2.line(image, (finalTarget[0], screenHeight), (finalTarget[0], 0), (255, 0, 0), 2)

    #     currentAngleError = finalTarget[1]
    #     # pushes vision target angle to network table


    # cv2.line(image, (round(centerX), screenHeight), (round(centerX), 0), (255, 255, 255), 2)

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
    #this checks that the area of the contour is greater than the image width divide by 2
    #And that the aspect ratio of the bounding rectangle (width / height) is close to 1 which 
    #is basically a circle however this would filter out 'tadpoles'
    print ("aspect ratio of ball: " + str(cntAspectRatio)) 
   # return (cntSize > (image_width / 2)) and (round(cntAspectRatio) > 1)
    return (cntSize > (image_width / 2)) and (cntAspectRatio > 0.75)


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

        cv2.ellipse(image, ellipse, (255, 255, 0), 3)
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
        print(img[y, x, 2], img[y, x, 1], img[y, x, 0], cv2.cvtColor(green,cv2.COLOR_BGR2HSV))


Driver = False
Tape = True
PowerCell = False
ControlPanel = False


img = images[0]

imgLength = len(images)

print("Hello Vision Team!")

while True:

    frame = img

    if Driver:

        processed = frame

    else:

        if Tape:

            threshold = threshold_video(lower_green, upper_green, frame)
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

    cv2.imshow("raw", img)
    cv2.imshow("processed", processed)
    #print("warped:" + str(warped))
    #cv2.imshow("warped", warped)
    cv2.setMouseCallback('raw', draw_circle)

    key = cv2.waitKey(0)
    print(key) 

    if key == 27:
        break

    currentImg += 1
    print(imgLength)

    if (currentImg == imgLength-1 ):
         currentImg = 0

    img = images[currentImg]





