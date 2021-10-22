# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  : Galonestar
# @file       :  zhen.py
# @Time    : 2021/10/22 8:20
# @Function:
# coding=gbk
import time
import cv2
cap = cv2.VideoCapture("4.avi")  # 读取文件
start_time = time.time()
counter = 0
# 获取视频宽度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*2)
# 获取视频高度
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*2)
fps = cap.get(cv2.CAP_PROP_FPS) #视频平均帧率
while (True):
    ret, frame = cap.read()
    # 键盘输入空格暂停，输入q退出
    key = cv2.waitKey(100) & 0xff
    if key == ord(" "):
        cv2.waitKey(0)
    if key == ord("q"):
        break
    counter += 1  # 计算帧数
    if (time.time() - start_time) != 0:  # 实时显示帧数
        cv2.putText(frame, "FPS {0}".format(float('%.1f' % (counter / (time.time() - start_time)))), (250, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    3)
        src = cv2.resize(frame, (frame_width // 2, frame_height // 2), interpolation=cv2.INTER_CUBIC)  # 窗口大小
        cv2.imshow('frame', src)
        print("FPS: ", counter / (time.time() - start_time))
        counter = 0
        start_time = time.time()
    time.sleep(1 / fps)  # 按原帧率播放
cap.release()
cv2.destroyAllWindows()