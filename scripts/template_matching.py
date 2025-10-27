# @Author: Zheng Pan
# @Emial: pan_zheng@nwpu.edu.cn
# Copyright (c) 2025 Zheng Pan
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from scripts.utils import cv_show
from scripts.process_template import prepocess_template
from scripts.preprocess_target import preprocess_target
import cv2 as cv
import numpy as np

def improved_template_matching(roi_binary, digits_dict, manual_centers=None,
                                size_number=None, size_dot=None):
    """
    针对段码数字的改进模板匹配方法
    
    参数:
        roi_binary: 预处理后的二值图像（白字黑底）
        digits_dict: 数字模板字典
        manual_centers: 手动提供的数字中心点列表
        
    返回:
        recognized_digits: 识别出的数字列表
        matched_image: 带有匹配结果的图像
        digit_regions: 每个数字的区域信息
    """
    # 复制图像用于绘制结果
    matched_image = cv.cvtColor(roi_binary, cv.COLOR_GRAY2BGR)

    # 查找所有轮廓（不进行形态学连接操作）
    contours, _ = cv.findContours(roi_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 创建所有轮廓的可视化
    all_contours_img = matched_image.copy()
    cv.drawContours(all_contours_img, contours, -1, (0, 255, 0), 1)
    cv_show('1. All Contours', all_contours_img, 0)

    # 打印轮廓数量信息
    print(f"检测到 {len(contours)} 个轮廓")

    # 存储轮廓属性
    contour_properties = []

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        aspect_ratio = w / float(h) if h != 0 else 0
        area = cv.contourArea(cnt)

        contour_properties.append({
            'cnt': cnt,
            'x': x, 'y': y, 'w': w, 'h': h,
            'area': area,
            'aspect_ratio': aspect_ratio,
            'center': (x + w/2, y + h/2)
        })

    # 使用手动提供的中心点进行分组
    if manual_centers is not None and len(manual_centers) == 4:
        print("使用手动提供的中心点进行分组")
        digit_groups = [[] for _ in range(len(manual_centers))]

        for props in contour_properties:
            cx, cy = props['center']
            min_dist = float('inf')
            group_idx = 0

            # 找到最近的数字中心
            for i, (center_x, center_y) in enumerate(manual_centers):
                dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    group_idx = i

            digit_groups[group_idx].append(props)
    else:
        print("错误: 需要提供4个手动中心点")
        return [], matched_image, []

    # 可视化分组结果
    grouped_image = matched_image.copy()
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]

    # 绘制手动中心点
    for i, (cx, cy) in enumerate(manual_centers):
        cv.circle(grouped_image, (int(cx), int(cy)), 1, colors[i], -1)
        # cv.putText(grouped_image, f"Center {i}", (int(cx)+5, int(cy)), 
        #           cv.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1)
    cv_show('Grouped Contours with Manual Centers', grouped_image, 0)

    for i, group in enumerate(digit_groups):
        color = colors[i % len(colors)]
        for props in group:
            x, y, w, h = props['x'], props['y'], props['w'], props['h']
            cv.rectangle(grouped_image, (x, y), (x+w, y+h), color, 1)
            
        cv_show('2. Grouped Contours with Manual Centers', grouped_image, 0)

        # 计算组的边界框
        if group:
            if size_number is not None:
                # pass
                cx, cy = manual_centers[i]
                w_num, h_num = int(size_number[0]), int(size_number[1])
                min_x = int(cx - w_num / 2)
                min_y = int(cy - h_num / 2)
                max_x = min_x + w_num
                max_y = min_y + h_num
                # 限制在图像范围内
                min_x = max(0, min_x)
                min_y = max(0, min_y)
                max_x = min(grouped_image.shape[1], max_x)
                max_y = min(grouped_image.shape[0], max_y)
                cv.rectangle(grouped_image, (min_x, min_y), (max_x, max_y), color, 1)
            else:
                min_x = min(props['x'] for props in group)
                max_x = max(props['x'] + props['w'] for props in group)
                min_y = min(props['y'] for props in group)
                max_y = max(props['y'] + props['h'] for props in group)
                cv.rectangle(grouped_image, (min_x, min_y), (max_x, max_y), color, 1)
                # cv.putText(grouped_image, f"Group {i}", (min_x, min_y-5), 
                #         cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            print(f"Group {i} 边界框: x={min_x}, y={min_y}, w={max_x - min_x}, h={max_y - min_y}")

    cv_show('2. Grouped Contours with Manual Centers', grouped_image, 0)

    # 为每个数字创建完整的二值图像
    digit_regions = []
    recognized_digits = []
    confidence_scores = []
    
    for i, group in enumerate(digit_groups):
        if not group:
            print(f"警告: 第 {i+1} 个数字没有检测到任何笔画段")
            recognized_digits.append(None)
            confidence_scores.append(0)
            continue
        

        # 计算数字区域
        if size_number is not None:
            # 使用手动圆心和给定的 size_number (w,h) 来提取固定大小的数字区域
            cx, cy = manual_centers[i]
            w_num, h_num = int(size_number[0]), int(size_number[1])
            min_x = int(cx - w_num / 2)
            min_y = int(cy - h_num / 2)
            max_x = min_x + w_num
            max_y = min_y + h_num
        else:
            min_x = min(props['x'] for props in group)
            max_x = max(props['x'] + props['w'] for props in group)
            min_y = min(props['y'] for props in group)
            max_y = max(props['y'] + props['h'] for props in group)
        
        # 确保区域在图像范围内
        min_x, min_y = max(0, min_x), max(0, min_y)
        max_x, max_y = min(roi_binary.shape[1], max_x), min(roi_binary.shape[0], max_y)
        
        # 提取数字区域
        digit_region = roi_binary[min_y:max_y, min_x:max_x]
        
        # 调整大小以匹配模板
        digit_resized = cv.resize(digit_region, (57, 88))
        
        # 存储区域信息
        digit_regions.append({
            'region': digit_region,
            'resized': digit_resized,
            'bbox': (min_x, min_y, max_x-min_x, max_y-min_y)
        })
        
        # 显示每个数字的区域
        cv_show(f'3. Digit {i} Region', digit_resized, 500)
        
        # 模板匹配
        best_match = None
        best_score = -float('inf')
        
        for digit, template in digits_dict.items():
            # 确保尺寸匹配
            if digit_resized.shape != template.shape:
                digit_resized_resized = cv.resize(digit_resized, (template.shape[1], template.shape[0]))
            else:
                digit_resized_resized = digit_resized
            
            # 模板匹配
            result = cv.matchTemplate(digit_resized_resized, template, cv.TM_CCOEFF_NORMED)
            _, score, _, _ = cv.minMaxLoc(result)
            
            # 更新最佳匹配
            if score > best_score:
                best_score = score
                best_match = digit
        
        recognized_digits.append(best_match)
        confidence_scores.append(best_score)
        
        # 在图像上绘制结果
        cv.rectangle(matched_image, (min_x, min_y), (max_x, max_y), colors[i], 2)
        cv.putText(matched_image, f"{best_match}({best_score:.2f})", 
                  (min_x, min_y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
        
        print(f"数字 {i+1}: 识别为 {best_match}, 置信度: {best_score:.2f}")
    
    cv_show('4. Final Matching Result', matched_image, 0)
    
    return recognized_digits, matched_image, digit_regions


if __name__ == "__main__":

    img = cv.imread("template/Segment_digital_tube_number_with_dot.png")
    digit_groups, digits_dict = prepocess_template(img)

    recorder_image = cv.imread("images/test_1024_frame_002090.jpg")
    box = [633, 225, 75, 37]
    angle = -4
    roi_binary = preprocess_target(recorder_image, bbox=box, angle=angle, threshold=160)


    print("\n尝试改进的模板匹配...")
    manual_centers = [(12, 26), (37, 26), (62, 26), (88, 26)]   # 长宽已插值为100x50
    # 长宽已插值为100x50, size_number=(23,40)
    recognized_digits_improved, matched_image_improved, digit_regions = improved_template_matching(
        roi_binary, digits_dict, manual_centers=manual_centers, size_number=(23,40))