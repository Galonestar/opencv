# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  : Galonestar
# @file       :  xmletree.py
# @Time    : 2021/10/22 9:18
# @Function:
"""
根据xml标签裁剪目标
1.文件夹中可以有除了图片格式或标签格式的文件，有判断可以忽略其他文件
2.图片未标注没有对应的xml文件自动忽略不会报错
3.同一个图片有多个真实框，会按名称_0,名称_1 …… 排列
4.图片本身的命名无规律会按3中的命名规则加上名称后缀重新命名。
"""

import cv2
import xml.etree.ElementTree as ET
import os

img_path = r'E:\DJI-ROCO\robomaster_Final Tournament/image/'  # 图片路径
xml_path = r'E:\DJI-ROCO\robomaster_Final Tournament/image_annotation/'  # 标签路径
obj_img_path = r'E:\DJI-ROCO\robomaster_Final Tournament/obj/'  # 目标裁剪图片存放路径
for img_file in os.listdir(img_path):  # 遍历图片文件夹
    """    
        if i == 100:
        i += 1
        break"""
    if img_file[-4:] in ['.bmp', '.jpg', '.png']:  # 判断文件是否为图片格式
        img_filename = os.path.join(img_path, img_file)  # 将图片路径与图片名进行拼接
        img_cv = cv2.imread(img_filename)  # 读取图片

        img_name = (os.path.splitext(img_file)[0])  # 分割出图片名，如“000.png” 图片名为“000”
        xml_name = xml_path + '\\' + '%s.xml' % img_name  # 利用标签路径、图片名、xml后缀拼接出完整的标签路径名

        if os.path.exists(xml_name):  # 判断与图片同名的标签是否存在，因为图片不一定每张都打标
            root = ET.parse(xml_name).getroot()  # 利用ET读取xml文件
            count = 0  # 目标框个数统计，防止目标文件覆盖
            for obj in root.iter('object'):  # 遍历所有目标框
                name = obj.find('name').text  # 获取目标框名称，即label名

                xmlbox = obj.find('bndbox')  # 找到框目标
                x0 = xmlbox.find('xmin').text  # 将框目标的四个顶点坐标取出
                y0 = xmlbox.find('ymin').text
                x1 = xmlbox.find('xmax').text
                y1 = xmlbox.find('ymax').text

                obj_img = img_cv[int(float(y0)):int(float(y1)), int(float(x0)):int(float(x1))]  # cv2裁剪出目标框中的图片
                if obj_img.all is not None:
                    cv2.imwrite(obj_img_path + '\\' + '%s_%s' % (img_name, count) + '.jpg', obj_img)  # 保存裁剪图片
                count += 1  # 目标框统计值自增1

print("裁剪完成！")