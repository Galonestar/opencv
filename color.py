
# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  : Galonestar
# @file       :  color.py
# @Time    : 2021/10/22 10:27
# @Function:

import cv2
import numpy as np

ball_color = 'red'

color_dist = {'red': {'Lower': np.array([0, 60, 60]), 'Upper': np.array([10, 255, 255])},
              'blue': {'Lower': np.array([100, 80, 46]), 'Upper': np.array([124, 255, 255])},
              'green': {'Lower': np.array([35, 43, 35]), 'Upper': np.array([90, 255, 255])},
              }

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow('camera', cv2.WINDOW_AUTOSIZE)
cv2.ocl.setUseOpenCL(False)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        if frame is None:
            print("无画面")

        else:
            gs_frame = cv2.GaussianBlur(frame, (5, 5), 0)                     # 高斯模糊
            hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)                 # 转化成HSV图像
            erode_hsv = cv2.erode(hsv, None, iterations=2)                   # 腐蚀 粗的变细
            inRange_hsv = cv2.inRange(erode_hsv, color_dist[ball_color]['Lower'], color_dist[ball_color]['Upper'])
            cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            c = max(list(cnts), key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            cv2.drawContours(frame, [np.int0(box)], -1, (0, 255, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = 'red'
            cv2.putText(frame, text, (100, 200), font, 2, (0, 0, 255), 1)

            cv2.imshow('camera', frame)
            cv2.waitKey(10)
            if 0xff == ord(" "):
                cv2.waitKey(0)
            if 0xff == ord("q"):
                break
    else:
        print("无法读取摄像头！")
cap.release()
cv2.destroyAllWindows()