# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  : Galonestar
# @file       :  polygon.py
# @Time    : 2021/10/22 8:44
# @Function:

import numpy as np
import cv2

def get_polygon_contours(self, img, img_back):
    img = np.copy(img)
    dif = np.array(img, dtype=np.int16)
    dif = np.abs(dif - img_back)
    dif = np.array(dif, dtype=np.uint8)  # get different

    gray = cv2.cvtColor(dif, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, self.min_thresh, 255, 0)
    thresh = cv2.blur(thresh, (self.thresh_blur, self.thresh_blur))

    if np.max(thresh) == 0:  # have not different
        contours = []  # 空列表在Python的逻辑判断中，是False
    else:
        thresh, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # hulls = [cv2.convexHull(cnt) for cnt, hie in zip(contours, hierarchy[0]) if hie[2] == -1]
        # hulls = [hull for hull in hulls if cv2.arcLength(hull, True) > self.min_hull_len]
        # contours = hulls

        approxs = [cv2.approxPolyDP(cnt, self.min_side_len, True) for cnt in contours]
        approxs = [approx for approx in approxs
                   if len(approx) > self.min_side_num and cv2.arcLength(approx, True) > self.min_poly_len]
        contours = approxs
    return contours