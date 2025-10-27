# @Author: Zheng Pan
# @Emial: pan_zheng@nwpu.edu.cn
# Copyright (c) 2025 Zheng Pan
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from scripts.convert_video_2_jpg import video_to_frames
from scripts.utils import cv_show, my_resize
from scripts.preprocess_target import preprocess_target
# from scripts import sort_contours, point_in_rect, my_resize  # noqa: F401
from scripts.process_template import prepocess_template
import cv2 as cv
import numpy as np
from scripts.template_matching import improved_template_matching

import os
import glob
import shutil


def clear_current_folder():
    current_dir = "output_frames"    # 获取当前工作目录
    
    for filename in os.listdir(current_dir):
        file_path = os.path.join(current_dir, filename)
        
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或符号链接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 递归删除子文件夹
        except Exception as e:
            print(f'删除 {file_path} 时出错: {e}')
    
    print("当前文件夹内容已清空")


def main():

    """ convert original data.mp4 to .png """
    clear_current_folder()
    # video_path = "test_record/1437485690.mp4"

    video_path = "test_record/turbojet_test_2025-1024-1413.mp4"
    output_folder = "output_frames"  # 输出文件夹名称
    video_to_frames(video_path, output_folder, start_time=0, end_time=None)

    """ extract number contour from template """
    img = cv.imread("template/Segment_digital_tube_number_with_dot.png")
    # cv_show('template', img, 2)

    # digit_groups(list)储存数字和轮廓, digits_dict(dict)存储数字(key))和对应的模板图像(value)
    digit_groups, digits_dict = prepocess_template(img)
    # cv_show(f'Digit {5} Template', digits_dict[5], 0)
    # cv_show(f'Digit {9} Template', digits_dict[9], 0)
    # cv_show(f'Dot', digits_dict['.'], 0)

    print("extract number contour from template successfully")

    # 初始化卷积核
    # rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 3))
    # sqKernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

    # return
    # tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, rectKernel)     # tophat可以突出图片中明亮的区域，过滤掉较暗的部分
    # cv_show('tophat', tophat)

    # single test
    """ preprocess target image to get roi_binary """
    # recorder_image = cv.imread("output_frames/frame_000690.jpg")

    box = [633, 225, 75, 37]
    angle = -4
    # roi_binary = preprocess_target(recorder_image, bbox=box, angle=angle, threshold=160)
    # cv_show('Processed ROI Binary', roi_binary)


    """template matching"""
    manual_centers = [(12, 26), (37, 26), (62, 26), (88, 26)]   # 长宽已插值为100x50
    # manual_centers = [(12, 26), (36, 26), (61, 26), (88, 26)]   # 长宽已插值为100x50
    # 长宽已插值为100x50, size_number=(23,40)
    roiSize_of_digital_number = (22, 40)
    # recognized_digits_improved, matched_image, digit_regions = improved_template_matching(
    #     roi_binary, digits_dict, manual_centers=manual_centers, size_number=roiSize_of_digital_number)
    
    # print(recognized_digits_improved)

    # multi image test

    input_folder = "output_frames"
    patterns = ['*.jpg', '*.jpeg', '*.png']
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(input_folder, p)))
    files = sorted(files)

    for idx, fp in enumerate(files, 1):
        fname = os.path.basename(fp)
        stem, _ = os.path.splitext(fname)
        print(f"[{idx}/{len(files)}] Processing {fname} ...")
        recorder_image = cv.imread(fp)
        if recorder_image is None:
            print(f"  Failed to read {fp}, skip.")
            continue

        roi_binary = preprocess_target(recorder_image, bbox=box, angle=angle, threshold=165)

        try:
            recognized_digits_improved, matched_image_improved, digit_regions = improved_template_matching(
                roi_binary, digits_dict, manual_centers=manual_centers, size_number=roiSize_of_digital_number)
        except Exception as e:
            print(f"  Error running improved_template_matching on {fname}: {e}")
            continue

        if matched_image_improved is None:
            print(f"  No matched_image returned for {fname}, skip saving.")
            continue

        out_name = f"{stem}_matched.png"
        out_path = os.path.join(input_folder, out_name)
        cv.imwrite(out_path, matched_image_improved)
        print(f"  Saved matched image to {out_path}")



    print("done")


if __name__ == "__main__":
    main()
