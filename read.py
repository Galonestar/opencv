#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/10/21 下午4:56
# @Author : Galonestar
# @Version : V 1.0
# @File : read.py
# @desc :

import cv2
import sys


def run_read(camera_name):
    video_stream_path = camera_name
    if camera_name == 999:
        video_stream_path = "4.avi"  # local camera (e.g. the front camera of laptop)
    cap = cv2.VideoCapture(video_stream_path)

    while cap.isOpened():
        is_opened, frame = cap.read()
        cv2.waitKey(50)
        cv2.imshow(str(camera_name), frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_read(0)
