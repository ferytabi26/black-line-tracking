#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# After the colour identification, execute the action group

import cv2
import numpy as np
import sys
import time
import threading
import math
from cv_ImgAddText import *
# import Serial_Servo_Running as SSR
# import signal
# import PWMServo

print('''
**********************************************************
*****Line following: detect the black line through the camera ato let the robot move along the black line*******
**********************************************************
----------------------------------------------------------
Official website:http://www.lobot-robot.com/pc/index/index
Online mall:https://lobot-zone.taobao.com/
----------------------------------------------------------
Version: --V3.0  2019/08/10
----------------------------------------------------------
''')
# PWMServo.setServo(1, 500, 500)
# PWMServo.setServo(2, 1500, 500)
# SSR.running_action_group('0', 1)

debug = True

# go_straight = '1'
# turn_left   = 'turn_right'
# turn_right  = 'turn_left'
# stand1      = 'stand_lrtog'
# stand2      = 'stand_gtolr'
ori_width  =  int(4*160)#Original image640x480
ori_height =  int(3*160)

line_color     = (255, 0, 0)#Draw the color of the wireframe when the image displays
line_thickness = 2         #Draw the thickness of the wireframe when the image displays

resolution = str(ori_width) + "x" + str(ori_height)

print('''
--The program is running correctly......
--Resolution:{0}                                                                                         
'''.format(resolution))

class Point(object):
    x = 0
    y = 0

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class Line(object):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

def GetCrossAngle(l1, l2):
    '''
    Calculate the angle between the two lines
    :param l1:
    :param l2:
    :return:
    '''
    arr_0 = np.array([(l1.p2.x - l1.p1.x), (l1.p2.y - l1.p1.y)])
    arr_1 = np.array([(l2.p2.x - l2.p1.x), (l2.p2.y - l2.p1.y)])
    cos_value = (float(arr_0.dot(arr_1)) / (np.sqrt(arr_0.dot(arr_0)) * np.sqrt(arr_1.dot(arr_1))))   # Caution: switch to floating- point calculation
    return np.arccos(cos_value) * (180/np.pi)

#Mapping function
def leMap(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

#Draw the circle and set parameters (Display text image, x coordinates, y coordinates, radius, width of processed image, height of image processed, color (optional) and size (optional))
def picture_circle(orgimage, x, y, r, resize_w, resize_h, l_c = line_color, l_t = line_thickness):
    global ori_width
    global ori_height
    
    x = int(leMap(x, 0, resize_w,  0, ori_width))
    y = int(leMap(y, 0, resize_h,  0, ori_height))
    r = int(leMap(r, 0, resize_w,  0, ori_width))   
    cv2.circle(orgimage, (x, y), r, l_c, l_t)

#Detect and return the maximum area
def getAreaMaxContour(contours,area=1):
        contour_area_max = 0
        area_max_contour = None

        for c in contours :
            contour_area_temp = math.fabs(cv2.contourArea(c))
            if contour_area_temp > contour_area_max : 
                contour_area_max = contour_area_temp
                if contour_area_temp > area:#The area greater than 1
                    area_max_contour = c
        return area_max_contour


# stream = "http://127.0.0.1:8080/?action=stream?dummy=param.mjpg"
cap = cv2.VideoCapture(0)
Running = True
orgFrame = None
ret = False
def get_image():
    global orgFrame
    global ret
    global Running
    global cap
    while True:
        if Running:
            if cap.isOpened():
                ret, orgFrame = cap.read()
            else:
                ret = False
                time.sleep(0.01)
        else:
            time.sleep(0.01)

# Display image thread
th1 = threading.Thread(target=get_image)
th1.setDaemon(True)     # Set the background thread, which defaults is "False", and if is set to "True", the thread doesn't have to wait for the sub-threads
th1.start()

roi = [ # [ROI, weight]
        (0,  40,  0, 160, 0.5), 
        (40, 80,  0, 160, 0.3), 
        (80, 120,  0, 160, 0.2)
       ]

angle = 0
get_line = False
deflection_angle = 0
def Tracing(orgimage, r_w, r_h, r = roi, l_c = line_color, l_t = line_thickness):
    global ori_width, ori_height
    global img_center_x, img_center_y
    global deflection_angle, angle
    global get_line
    #Shrink image, accelerate processing speed
    orgframe = cv2.resize(orgimage, (r_w, r_h), interpolation = cv2.INTER_LINEAR)
    orgframe = cv2.cvtColor(orgframe, cv2.COLOR_BGR2GRAY)#Convert to grayscale image
    orgframe = cv2.GaussianBlur(orgframe, (3,3), 0)#Gaussian blur, denoising
    _, Imask = cv2.threshold(orgframe, 50, 255, cv2.THRESH_BINARY_INV)#Binarization
    Imask = cv2.erode(Imask, None, iterations=2)
    Imask = cv2.dilate(Imask, np.ones((3, 3), np.uint8), iterations=2)
    centroid_x_sum = 0
    area_sum = 0
    n = 0
    weight_sum = 0
    center_ = []
    max_area = 0
    for r in roi:
        n += 1
        blobs = Imask[r[0]:r[1], r[2]:r[3]]
        cnts, _ = cv2.findContours(blobs , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)#Find out all the outline
        cnt_large  = getAreaMaxContour(cnts)#Find the outline of the largest area
        if cnt_large is not None:
            rect = cv2.minAreaRect(cnt_large)#The smallest outer rectangle
            box = np.int0(cv2.boxPoints(rect))#The four vertices of the smallest enclosing rectangle
            box[0, 1], box[1, 1], box[2, 1], box[3, 1] = box[0, 1] + (n - 1)*r_w/4, box[1, 1] + (n - 1)*r_w/4, box[2, 1] + (n - 1)*r_w/4, box[3, 1] + (n - 1)*r_w/4
            box[1, 0] = int(leMap(box[1, 0], 0, r_w, 0, ori_width))
            box[1, 1] = int(leMap(box[1, 1], 0, r_h, 0, ori_height))
            box[3, 0] = int(leMap(box[3, 0], 0, r_w, 0, ori_width))
            box[3, 1] = int(leMap(box[3, 1], 0, r_h, 0, ori_height))
            box[0, 0] = int(leMap(box[0, 0], 0, r_w, 0, ori_width))
            box[0, 1] = int(leMap(box[0, 1], 0, r_h, 0, ori_height))
            box[2, 0] = int(leMap(box[2, 0], 0, r_w, 0, ori_width))
            box[2, 1] = int(leMap(box[2, 1], 0, r_h, 0, ori_height))
            pt1_x, pt1_y = box[0, 0], box[0, 1]
            pt3_x, pt3_y = box[2, 0], box[2, 1]
            area = cv2.contourArea(box)
            cv2.drawContours(frame, [box], -1, (0,0,255,255), 2)#Draw a rectangle composed of four points            
            center_x, center_y = (pt1_x + pt3_x) / 2, (pt1_y + pt3_y) / 2#Central point
            center_.append([center_x,center_y])            
            cv2.circle(frame, (int(center_x), int(center_y)), 10, (0,0,255), -1)#Draw the central point
            centroid_x_sum += center_x * r[4]
            weight_sum += r[4]

    if weight_sum is not 0:
        center_x_pos = centroid_x_sum / weight_sum
        #median formula
        deflection_angle = 0.0
        deflection_angle = -math.atan((center_x_pos - img_center_x/2)/(img_center_y/2))
        deflection_angle = deflection_angle*180.0/math.pi
        #print(center_x_pos)
         #Draw a cross in the center of the frame
        #cv2.line(orgimage, (img_center_x/2, img_center_y), (int(center_x_pos), img_center_y/2), l_c, l_t)         
    get_line = True
    
state1 = 0 
def move():
    global deflection_angle, angle
    global get_line, state1
    global go_straight,turn_left,turn_right, stand
    while True:
        if get_line:
            get_line = False
            if -25 <= deflection_angle <= 25:
                if state1 != 1:
                    # SSR.run_ActionGroup(stand1, 1)
                    print('stand1')
                    time.sleep(0.1)
                # state1 = 1
                print('gostraigh')
                # SSR.run_ActionGroup(go_straight, 1)           
            elif deflection_angle > 25:
                if state1 == 1:
                    print('gostraigh2')
                    # SSR.run_ActionGroup(stand2, 1)
                    time.sleep(0.1)
                # state1 = 2
                print('turnleft')
                # SSR.run_ActionGroup(turn_left, 1)
            elif deflection_angle < -25:
                if state1 == 1:
                    print('stand2')
                    # SSR.run_ActionGroup(stand2, 1)
                    time.sleep(0.1)
                # state1 = 3
                print('turn rigth')
                # SSR.run_ActionGroup(turn_right, 1)
        else:
            time.sleep(0.01)
      
th2 = threading.Thread(target=move)
th2.setDaemon(True)     # Set the background thread, which defaults is "False", and if is set to "True", the thread doesn't have to wait for the sub-threads
th2.start()    

while True:
    if orgFrame is not None and ret:
        if Running:
            t1 = cv2.getTickCount()
            img_center_x = orgFrame.shape[:2][1]
            img_center_y = orgFrame.shape[:2][0]
            frame = orgFrame.copy()
            Tracing(orgFrame, 160, 120)    
            t2 = cv2.getTickCount()
            time_r = (t2 - t1) / cv2.getTickFrequency()
            fps = 1.0/time_r
            if debug == 1:#In the debug mode
                frame = cv2ImgAddText(frame, "Line following", 10, 10, textColor=(0, 0, 0), textSize=20)
                cv2.putText(frame, "FPS:" + str(int(fps)),
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)#(0, 0, 255)BGR
                cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)#Display frame name
                #cv2.moveWindow('frame', img_center_x, 100)#Display frame position
                cv2.imshow('frame', frame) #Display image
                cv2.waitKey(1)
        else:
            time.sleep(0.01)
    else:
        time.sleep(0.01)
cv2.destroyAllWindows()
