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
from scripts.template_matching import improved_template_matching, find_dot

import os
import glob
import shutil

DEBUG=0

def clear_output_folder():
    current_dir = "output_frames"    # 获取储存目录
    
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


def recognized_digits_number(recognized_digits, dot_flags):
    """
    根据识别结果和小数点标志，生成最终数字。
    如果有 None,返回 None;否则返回 float 或 int。
    """
    if any(d is None for d in recognized_digits):
        return None  # 或 "识别失败"
    # 拼接字符串，插入小数点
    result = ""

    for i, digit in enumerate(recognized_digits):
        result += str(digit)

    num = int(result)
    if dot_flags[0]:
        num = num / 1000
    elif dot_flags[1]:
        num = num / 100
    return num


""" 主程序入口 """
import csv
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="test_record/turbojet_test_2025-1024-1413.mp4", help="Path of data files")
    parser.add_argument("--output", type=str, default="output/turbojet_test_data.csv", help="Path to save the data csv")
    args = parser.parse_args()
    

    """ Step.1. convert original data.mp4 to .png """
    clear_output_folder()

    video_path = args.data
    # video_path = "test_record/turbojet_test_2025-1024-1413.mp4"
    output_folder = "output_frames"  # 输出文件夹名称
    FPS = 29 # 我的小米手机录制的视频就是29fps
    frame_interval = 10
    video_to_frames(video_path, output_folder, start_time=0, end_time=None,frame_interval=frame_interval)

    """Step.2.  extract number contour from template """
    img = cv.imread("template/Segment_digital_tube_number_with_dot.png")

    digit_groups, digits_dict = prepocess_template(img)
    print("extract number contour from template successfully")


    """ Step.3. preprocess target image to get roi_binary """
    box = [633, 225, 75, 37]
    angle = -4

    """ Step.4. template matching"""
    manual_centers = [(12, 26), (37, 26), (62, 26), (88, 26)]   # 长宽已插值为100x50
    roiSize_of_digital_number = (22, 40)

    # outputfile = "turbojet_test_data.csv"
    outputfile = args.output
    header = ["time", "forces"]
    os.makedirs(os.path.dirname(outputfile), exist_ok=True)
    with open(outputfile, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    print(f"[INFO] CSV 文件已创建: {outputfile}")

    input_folder = "output_frames"
    patterns = ['*.jpg', '*.jpeg', '*.png']
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(input_folder, p)))
    files = sorted(files)

    for idx, fp in enumerate(files, 1):
        video_time = idx*frame_interval/FPS    # 计算时间

        fname = os.path.basename(fp)
        stem, _ = os.path.splitext(fname)
        print(f"[{idx}/{len(files)}] Processing {fname} ...")
        recorder_image = cv.imread(fp)
        recorder_image_copy = recorder_image.copy()
        
        if recorder_image is None:
            print(f"  Failed to read {fp}, skip.")
            continue

        roi_binary = preprocess_target(recorder_image, bbox=box, angle=angle, threshold=150)

        if DEBUG:
            cv_show('Processed ROI Binary', roi_binary, 0)

        try:
            recognized_digits, matched_image, digit_regions = improved_template_matching(
                roi_binary, digits_dict, manual_centers=manual_centers, size_number=roiSize_of_digital_number)
        except Exception as e:
            print(f"  Error running improved_template_matching on {fname}: {e}")
            continue

        if matched_image is None:
            print(f"  No matched_image returned for {fname}, skip saving.")
            continue
        
        dot_flags = find_dot(roi_binary, manual_dot_centers = [(24, 44), (48, 44)], size_dot=(2,2))
        number = recognized_digits_number(recognized_digits, dot_flags)
        print(f"recognized_digits_number={number}")

        # 记录数字和时间到图像
        cv.putText(recorder_image_copy, f"number: {number}", 
                  (30, 100), cv.FONT_HERSHEY_SIMPLEX, 3, [0,0,255], 3)
        cv.putText(recorder_image_copy, f"video time {video_time}", 
                  (30, 200), cv.FONT_HERSHEY_SIMPLEX, 3, [0,0,255], 3)
        if DEBUG:
            cv_show("recorder_image",recorder_image_copy)

        row = [f"{video_time:.2f}", f"{number if number is not None else '识别失败'}"]
        # Append to CSV
        with open(outputfile, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        out_name = f"{stem}_matched.png"
        out_path = os.path.join(input_folder, out_name)

        cv.imwrite(out_path, matched_image)
        print(f"  Saved matched image to {out_path}")



    print("done")


if __name__ == "__main__":
    main()
