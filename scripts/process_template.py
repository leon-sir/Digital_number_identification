# @Author: Zheng Pan
# @Emial: pan_zheng@nwpu.edu.cn
# Copyright (c) 2025 Zheng Pan
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cv2 as cv
import numpy as np
from scripts.utils import cv_show, point_in_rect

# RoI: [x,y,w,h], (x, y): center of the number

DEBUG = 0

def prepocess_template(img):
    """
    处理七位段码数字,模板从1-9,0
    Return: digit_groups(list)储存数字和轮廓, 
    digits_dict(dict)存储数字(key))和对应的模板图像(value)
    
    """

    manual_bboxes = [
        [123, 167, 171, 222],  # 数字 1
        [288, 167, 171, 222],  # 数字 2
        [453, 167, 171, 222],  # 数字 3
        [618, 167, 171, 222],  # 数字 4
        [782, 167, 171, 222],  # 数字 5
        [121, 426, 171, 222],  # 数字 6
        [292, 420, 171, 222],  # 数字 7
        [452, 426, 171, 222],  # 数字 8
        [617, 426, 171, 223],  # 数字 9
        [782, 426, 171, 223],  # 数字 0
    ]

    ref = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if DEBUG:
        cv_show('1. gry template', ref, 2)    # 显示灰度图

    _, ref_binary = cv.threshold(ref, 80, 255, cv.THRESH_BINARY)
    if DEBUG:
        cv_show('2. Binary Image (After Thresholding)', ref_binary, 2) # 显示二值化后的效果，此时数字应为纯白，背景为纯黑

    # 对模板进行轮廓检测，得到轮廓信息
    refCnts, hierarchy = cv.findContours(ref_binary.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.drawContours(img, refCnts, -1, (0, 255, 255), 2)  # 第一个参数为目标图像
    if DEBUG:
        cv_show('3 Contours on Image ', img, 0)
    print("处理模板ing...")
    print(f"检测到 {len(refCnts)} 个轮廓")  # 1-9,0 共49笔

    # 使用手动边界框分组轮廓
    digit_groups = [[] for _ in range(len(manual_bboxes))]

    for cnt in refCnts:
        # 计算轮廓的中心点
        M = cv.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # 检查中心点属于哪个数字的边界框
            for i, bbox in enumerate(manual_bboxes):
                if point_in_rect((cx, cy), bbox):
                    digit_groups[i].append(cnt)
                    break

    # 可视化分组结果
    img_with_groups = img.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            ]

    # 创建数字模板字典
    digits_dict = {}

    for i, (group, bbox) in enumerate(zip(digit_groups, manual_bboxes)):
        color = colors[i % len(colors)]

        # 绘制数字边界框
        x, y, w, h = bbox
        cv.rectangle(img_with_groups, 
                     (int(x - w/2), int(y - h/2)), 
                     (int(x + w/2), int(y + h/2)), 
                     color, 2)

        # 绘制属于该数字的轮廓
        cv.drawContours(img_with_groups, group, -1, color, 1)

        # 根据索引确定标记内容
        if i < 9:
            label = str(i + 1)  # 数字1-9
        elif i == 9:
            label = '0'  # 数字0

        # 标记数字编号
        cv.putText(img_with_groups, label, 
                   (int(x - w/2) + 5, int(y - h/2) + 15), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 为每个数字创建模板
        size_number = (57, 88)
        angle = 1.5
        center = (size_number[0]//2, size_number[1]//2)  # 图像中心作为旋转中心
        M = cv.getRotationMatrix2D(center, angle, 1.0)

        # rotated_full_image = cv.warpAffine(img, M, [img.shape[1], img.shape[0]])

        if group:
            # 创建一个与边界框相同大小的空白图像
            digit_template = np.zeros((int(h), int(w)), dtype=np.uint8)

            # 将轮廓绘制到模板上
            for cnt in group:
                # 调整轮廓坐标到模板坐标系
                offset_cnt = cnt - [int(x - w/2), int(y - h/2)]
                cv.drawContours(digit_template, [offset_cnt], -1, 255, -1)  # 填充轮廓


            digit_template = cv.resize(digit_template, size_number)
            digit_template = cv.warpAffine(digit_template, M, [digit_template.shape[1], digit_template.shape[0]])
            # 存储模板，对小数点使用特殊键值'.'
            if i < 9:
                digits_dict[i + 1] = digit_template
            elif i == 9:
                digits_dict[0] = digit_template


    print(f"数字resize为了{size_number} \n" \
            "当前给定区域size, template_matching时把数字区域也照这个resizing")

    if DEBUG:
        cv_show('4. Grouped Contours with Manual BBoxes', img_with_groups, 0)     # 显示分组结果

    print("\n 模板数字分组结果统计:")    
    for i, group in enumerate(digit_groups):
        if i < 9:
            label = f"段码数字 {i + 1}"
        elif i == 9:
            label = "段码数字 0"
        print(f"{label}: {len(group)} 个轮廓")

    return digit_groups, digits_dict


def prepocess_template_with_dot(img):

    """
    处理七位段码数字,模板从1-9, 0, 和小数点
    """

    manual_bboxes = [
        [123, 167, 171, 222],  # 数字 1
        [288, 167, 171, 222],  # 数字 2
        [453, 167, 171, 222],  # 数字 3
        [618, 167, 171, 222],  # 数字 4
        [782, 167, 171, 222],  # 数字 5
        [121, 426, 171, 222],  # 数字 6
        [289, 426, 171, 222],  # 数字 7
        [452, 426, 171, 222],  # 数字 8
        [617, 426, 171, 223],  # 数字 9
        [782, 426, 171, 223],  # 数字 0
        [899, 520, 32, 32],    # 小数点
    ]

    ref = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv_show('1. gry template', ref, 2)    # 显示灰度图

    _, ref_binary = cv.threshold(ref, 80, 255, cv.THRESH_BINARY)
    # cv_show('2. Binary Image (After Thresholding)', ref_binary, 2) # 显示二值化后的效果，此时数字应为纯白，背景为纯黑

    # 对模板进行轮廓检测，得到轮廓信息
    refCnts, hierarchy = cv.findContours(ref_binary.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.drawContours(img, refCnts, -1, (0, 255, 255), 2)  # 第一个参数为目标图像
    # cv_show('3 Contours on Image ', img, 0)
    print("处理模板ing...")
    print(f"检测到 {len(refCnts)} 个轮廓")  # 1-9,0 共49笔

    # 使用手动边界框分组轮廓
    digit_groups = [[] for _ in range(len(manual_bboxes))]

    for cnt in refCnts:
        # 计算轮廓的中心点
        M = cv.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # 检查中心点属于哪个数字的边界框
            for i, bbox in enumerate(manual_bboxes):
                if point_in_rect((cx, cy), bbox):
                    digit_groups[i].append(cnt)
                    break

    # 可视化分组结果
    img_with_groups = img.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
              (0, 128, 128)]  # 新增小数点的颜色

    # 创建数字模板字典
    digits_dict = {}

    for i, (group, bbox) in enumerate(zip(digit_groups, manual_bboxes)):
        color = colors[i % len(colors)]

        # 绘制数字边界框
        x, y, w, h = bbox
        cv.rectangle(img_with_groups, 
                     (int(x - w/2), int(y - h/2)), 
                     (int(x + w/2), int(y + h/2)), 
                     color, 2)

        # 绘制属于该数字的轮廓
        cv.drawContours(img_with_groups, group, -1, color, 1)

        # 根据索引确定标记内容
        if i < 9:
            label = str(i + 1)  # 数字1-9
        elif i == 9:
            label = '0'  # 数字0
        else:
            label = '.'  # 小数点

        # 标记数字编号
        cv.putText(img_with_groups, label, 
                   (int(x - w/2) + 5, int(y - h/2) + 15), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 为每个数字创建模板
        size_number = (57, 88)
        size_dot = (32, 32)
        
        if group:
            # 创建一个与边界框相同大小的空白图像
            digit_template = np.zeros((int(h), int(w)), dtype=np.uint8)

            # 将轮廓绘制到模板上
            for cnt in group:
                # 调整轮廓坐标到模板坐标系
                offset_cnt = cnt - [int(x - w/2), int(y - h/2)]
                cv.drawContours(digit_template, [offset_cnt], -1, 255, -1)  # 填充轮廓

            # 根据是否为小数点调整模板大小
            if i == 10:  # 小数点使用原始尺寸
                digit_template = cv.resize(digit_template, size_dot)
            else:  # 数字使用标准尺寸
                digit_template = cv.resize(digit_template, size_number)

            # 存储模板，对小数点使用特殊键值'.'
            if i < 9:
                digits_dict[i + 1] = digit_template
            elif i == 9:
                digits_dict[0] = digit_template
            else:
                digits_dict['.'] = digit_template

    print(f"小数点resize为了{size_dot}, 数字resize为了{size_number} \n" \
            " 当前给定区域size, template_matching时把数字区域也照这个resizing")

    # cv_show('4. Grouped Contours with Manual BBoxes', img_with_groups, 0)     # 显示分组结果

    print("\n模板数字分组结果统计:")    
    for i, group in enumerate(digit_groups):
        if i < 9:
            label = f"段码数字 {i + 1}"
        elif i == 9:
            label = "段码数字 0"
        else:
            label = "小数点"
        print(f"{label}: {len(group)} 个轮廓")

    return digit_groups, digits_dict



if __name__ == "__main__":
    if 0:
        img = cv.imread("template/Segment_digital_tube_number_with_dot.png")
        # cv_show('template', img, 2)
        digit_groups, digits_dict = prepocess_template_with_dot(img)     
        cv_show(f'Digit {5} Template', digits_dict[5], 0)
        cv_show(f'Digit {9} Template', digits_dict[9], 0)
        cv_show(f'Dot', digits_dict['.'], 0)
    else:

        img = cv.imread("template/Segment_digital_tube_number.png")
        # cv_show('template', img, 2)
        digit_groups, digits_dict = prepocess_template(img)

        cv_show(f'Digit {5} Template', digits_dict[1], 0)
        cv_show(f'Digit {9} Template', digits_dict[7], 0)
