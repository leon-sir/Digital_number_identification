# @Author: Zheng Pan
# @Emial: pan_zheng@nwpu.edu.cn
# Copyright (c) 2025 Zheng Pan
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from scripts.utils import cv_show
import cv2 as cv


def preprocess_target(image, **kwargs):
    """
    处理目标图像，支持多种参数组合

    参数:
        image: 输入图像
        !!注意,返回的二值化图像w,h固定为100x50
        **kwargs: 可选参数，包括:
            - bbox: 边界框 [x, y, w, h]
            - angle: 旋转角度
            - threshold: 二值化阈值
        ** Return: roi_binary: 处理后的二值化图像
    """
    # 设置默认参数
    default_params = {
        'bbox': [262, 215, 86, 37],
        'angle': -9,
        'threshold': 170
    }
    # 更新默认参数
    params = {**default_params, **kwargs}

    # 记录仪边界框
    x, y, w, h = params['bbox']

    """ 旋转校正 旋转角度angle需要 人为设置"""
    angle = params['angle']
    center = (x, y)  # 记录仪RoI作为旋转中心

    M = cv.getRotationMatrix2D(center, angle, 1.0)
    print(image.shape)

    rotated_full_image = cv.warpAffine(image, M, [image.shape[1], image.shape[0]])
    # cv_show('Rotated Full Image', rotated_full_image)

    roi_image = rotated_full_image[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
    # cv_show('ROI After Rotation', roi_image)

    roi_gray = cv.cvtColor(roi_image, cv.COLOR_BGR2GRAY)
    # cv_show('gray', roi_gray)

    """ 边缘提取, threshold由屏幕亮度决定,背景色亮的需要使用 cv.THRESH_BINARY_INV """
    threshold = params['threshold']
    _, roi_binary = cv.threshold(roi_gray, threshold, 255, cv.THRESH_BINARY_INV)
    # cv_show('gray', roi_binary)
    roi_binary = cv.resize(roi_binary, (100, 50), interpolation=cv.INTER_NEAREST)
    # cv_show('gray', roi_binary)
    return roi_binary


if __name__ == "__main__":
    # recorder_image = cv.imread("images/test_old_frame_001340.jpg")
    # cv_show('Original Image', recorder_image)
    # box = [258, 245, 86, 37]
    # roi_binary = preprocess_target(recorder_image, bbox=box, angle=-15, threshold=140)
    # cv_show('Processed ROI Binary', roi_binary)

    recorder_image = cv.imread("images/test_1024_frame_002090.jpg")
    cv_show('Original Image', recorder_image)
    box = [633, 225, 74, 37]
    angle = -4
    roi_binary = preprocess_target(recorder_image, bbox=box, angle=angle, threshold=150)  # 160 
    cv_show('Processed ROI Binary', roi_binary)

    