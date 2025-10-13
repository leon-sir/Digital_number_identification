# @Author: Zheng Pan
# @Emial: pan_zheng@nwpu.edu.cn
# Copyright (c) 2025 Zheng Pan
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cv2 as cv
# import numpy as np
# import myutils


def cv_show(name, img, wait=0):		# 自定义的展示函数
    cv.imshow(name, img)
    cv.waitKey(wait*1000)  # seconds


# def sort_contours(cnts, method="left-to-right"):
#     reverse = False
#     i = 0
#     if method == "right-to-left" or method == "bottom-to-top":
#         reverse = True
#     if method == "top-to-bottom" or method == "bottom-to-top":
#         i = 1
#     boundingBoxes = [cv.boundingRect(c) for c in cnts]  # 用一个最小的矩形，把找到的形状包起来x,y,h,w
#     (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
#                                         key=lambda b: b[1][i], reverse=reverse))
#     return cnts, boundingBoxes  # 轮廓和boundingBoxess


def my_resize(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv.resize(image, dim, interpolation=inter)
    return resized


def sort_contours(cnts, rows=2, method="top-to-bottom-left-to-right"):
    # 获取所有轮廓的边界框
    boundingBoxes = [cv.boundingRect(c) for c in cnts]

    # 计算平均高度，用于行分组
    avg_height = sum([h for (x, y, w, h) in boundingBoxes]) / len(boundingBoxes)

    # 按y坐标排序（从上到下）
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][1]))

    # 分组到行
    rows_contours = []
    rows_boxes = []
    current_row = []
    current_row_boxes = []
    current_y = boundingBoxes[0][1]  # 第一个轮廓的y坐标

    for i, (cnt, (x, y, w, h)) in enumerate(zip(cnts, boundingBoxes)):
        # 如果当前轮廓与当前行的y坐标差异小于平均高度的一半，则认为在同一行
        if abs(y - current_y) <= avg_height / 2:
            current_row.append(cnt)
            current_row_boxes.append((x, y, w, h))
        else:
            # 新的一行开始
            rows_contours.append(current_row)
            rows_boxes.append(current_row_boxes)
            current_row = [cnt]
            current_row_boxes = [(x, y, w, h)]
            current_y = y

    # 添加最后一行
    if current_row:
        rows_contours.append(current_row)
        rows_boxes.append(current_row_boxes)

    # 确保只有指定行数
    if len(rows_contours) > rows:
        # 如果检测到的行数多于预期，合并最接近的行
        while len(rows_contours) > rows:
            # 找到最接近的两行
            min_diff = float('inf')
            merge_idx = 0
            for i in range(len(rows_contours)-1):
                y1 = rows_boxes[i][0][1]  # 第一行第一个轮廓的y坐标
                y2 = rows_boxes[i+1][0][1]  # 下一行第一个轮廓的y坐标
                if abs(y2 - y1) < min_diff:
                    min_diff = abs(y2 - y1)
                    merge_idx = i

            # 合并这两行
            rows_contours[merge_idx].extend(rows_contours[merge_idx+1])
            rows_boxes[merge_idx].extend(rows_boxes[merge_idx+1])
            del rows_contours[merge_idx+1]
            del rows_boxes[merge_idx+1]

    # 对每行内的轮廓按x坐标排序（从左到右）
    sorted_cnts = []
    sorted_boxes = []

    for i, (row_cnts, row_boxes) in enumerate(zip(rows_contours, rows_boxes)):
        # 对当前行按x坐标排序
        (sorted_row_cnts, sorted_row_boxes) = zip(*sorted(zip(row_cnts, row_boxes),
                                                         key=lambda b: b[1][0]))
        sorted_cnts.extend(sorted_row_cnts)
        sorted_boxes.extend(sorted_row_boxes)

    return sorted_cnts, sorted_boxes


def point_in_rect(point, rect):
    x, y = point
    rx, ry, rw, rh = rect
    return (rx - rw/2 <= x <= rx + rw/2) and (ry - rh/2 <= y <= ry + rh/2)
