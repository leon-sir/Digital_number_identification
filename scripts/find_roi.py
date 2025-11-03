# @Author: Zheng Pan
# @Emial: pan_zheng@nwpu.edu.cn
# Copyright (c) 2025 Zheng Pan
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import cv2 as cv
import argparse

# img = cv.imread("template/Segment_digital_tube_number.png")     # template
# img = cv.imread("output_frames/frame_000810.jpg")     # reader

points = []


def click_event(event, x, y, flags, param):

    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"点击坐标: ({x}, {y})")
        cv.circle(param, (x, y), 3, (0, 255, 0), -1)
        cv.imshow('Image', param)


def main():

    parser = argparse.ArgumentParser(description="手动标注图像坐标并输出边界框信息。")
    parser.add_argument(
        "--img",
        type=str,
        # required=True,
        default="output_frames/frame_007627.jpg",
        help="imput the image path. for example: python click_bbox.py --img template/Segment_digital_tube_number.png"
    )
    args = parser.parse_args()

    img = cv.imread(args.img)
    if img is None:
        raise FileNotFoundError(f"Can not find image: {args.img}")

    # 显示图像并等待点击
    cv.imshow('Image', img)
    
    cv.setMouseCallback('Image', click_event, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # 计算边界框
    if len(points) >= 2:
        x_min = min(p[0] for p in points)
        x_max = max(p[0] for p in points)
        y_min = min(p[1] for p in points)
        y_max = max(p[1] for p in points)

        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min

        print(f"边界框: [{cx}, {cy}, {w}, {h}]")


if __name__ == "__main__":
    main()
