# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import time
from my_bag import *
import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

if __name__ == '__main__':

    target_color = 'blue'

    color_dist = {'red': {'Lower': np.array([0, 60, 60]), 'Upper': np.array([6, 255, 255])},
                  'blue': {'Lower': np.array([100, 80, 46]), 'Upper': np.array([124, 255, 255])},
                  'gray': {'Lower': np.array([128, 150, 128]), 'Upper': np.array([169, 160, 169])},
                  }

    cap = cv2.VideoCapture('4.avi')
    cv2.namedWindow('camera', cv2.WINDOW_AUTOSIZE)
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
    kernel = np.ones((5, 5), np.uint8)
    background = None

    if cap.isOpened():
        print('Open')
    else:
        print('无素材')

    # 获取视频宽度
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 2)
    # 获取视频高度
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 2)
    size = (frame_width, frame_height)
    print('size:' + repr(size))

    cv2.ocl.setUseOpenCL(False)
    start_time = time.time()
    counter = 0
    fps = cap.get(cv2.CAP_PROP_FPS)  # 视频平均帧率

    while cap.isOpened():
        grabbed, frame = cap.read()

        # 图像处理部分
        # 降噪（模糊处理用来减少瑕疵点）
        frame = cv2.blur(frame, (3, 3))
        # 对帧进行预处理，先转灰度图，再进行高斯滤波。
        # 用高斯滤波进行模糊处理，进行处理的原因：每个输入的视频都会因自然震动、光照变化或者摄像头本身等原因而产生噪声。对噪声进行平滑是为了避免在运动和跟踪时将其检测出来。
        gray_lwpCV = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)

        frame_gs = cv2.GaussianBlur(frame, (5, 5), 0)  # 高斯模糊
        frame_hsv = cv2.cvtColor(frame_gs, cv2.COLOR_BGR2HSV)  # 转化成HSV图像
        frame_erode_hsv = cv2.erode(frame_hsv, None, iterations=2)  # 腐蚀 粗的变细
        frame_inRange_hsv = cv2.inRange(frame_erode_hsv, color_dist[target_color]['Lower'],
                                        color_dist[target_color]['Upper'])
        frame_fcc = cv2.findContours(frame_inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # 将第一帧设置为整个输入的背景
        if background is None:
            background = gray_lwpCV
            continue

        # 对于每个从背景之后读取的帧都会计算其与背景之间的差异，并得到一个差分图（different map）。
        # 还需要应用阈值来得到一幅黑白图像，并通过下面代码来膨胀（dilate）图像，从而对孔（hole）和缺陷（imperfection）进行归一化处理
        diff = cv2.absdiff(background, gray_lwpCV)
        diff = cv2.threshold(diff, 148, 255, cv2.THRESH_BINARY)[1]  # 二值化阈值处理
        diff = cv2.dilate(diff, es, iterations=2)  # 形态学膨胀

        # 显示矩形框1  运动监测
        contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)  # 该函数计算一幅图像中目标的轮廓

        for c1 in contours:
            if cv2.contourArea(c1) < 1500:
                # 对于矩形区域，只显示大于给定阈值的轮廓，所以一些微小的变化不会显示。对于光照不变和噪声低的摄像头可不设定轮廓最小尺寸的阈值
                continue
            (x, y, w, h) = cv2.boundingRect(c1)  # 该函数计算矩形的边界框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 显示矩形框2  装甲板监测 目标色块
        if frame_fcc is not None:
            fcc = max(frame_fcc, key=cv2.contourArea)
            rect = cv2.minAreaRect(fcc)
            box = cv2.boxPoints(rect)
            cv2.drawContours(frame, [np.int0(box)], -1, (0, 255, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = target_color
            box = np.int0(box)
            cv2.putText(frame, text, (np.min(box[:, 0]), np.max(box[:, 1])), font, 1, (0, 0, 255), 2)

        # 显示帧率
        counter += 1  # 计算帧数
        if (time.time() - start_time) != 0:  # 实时显示帧数
            cv2.putText(frame, "FPS {0}".format(float('%.1f' % (counter / (time.time() - start_time)))), (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                        3)
            print("FPS: ", counter / (time.time() - start_time))
            counter = 0
            start_time = time.time()

        # 文字显示 打击点坐标
        font = cv2.FONT_HERSHEY_SIMPLEX
        att_point = '('+str(np.min(box[:, 0]))+','+str(np.max(box[:, 1]))+')'
        cv2.putText(frame, att_point, (250, 50), font, 1, (0, 255, 0), 1)

        # 显示输出结果
        cv2.imshow('camera', frame)
        cv2.waitKey(1)

        # 键盘输入空格暂停，输入q退出
        key = cv2.waitKey(100) & 0xff
        if key == ord(" "):
            cv2.waitKey(0)
        if key == ord("q"):
            break
        if key == ord("p"):
            cv2.waitKey(1000000)
        time.sleep(1 / fps)  # 按原帧率播放

    cap.release()
    cv2.destroyAllWindows()
