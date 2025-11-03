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


DEBUG=0


def improved_template_matching(roi_binary, digits_dict, manual_centers=None,
                                size_number=None):
    """
    针对段码数字的改进模板匹配方法
    
    参数:
        roi_binary: 预处理后的二值图像（白字黑底）
        digits_dict: 数字模板字典
        manual_centers: 手动提供的数字中心点列表
        size_number: 每个数字的固定大小 (宽, 高)，如果为 None 则根据轮廓计算
        size_dot: 小数点的固定大小 (宽, 高)，如果为 None 则根据轮廓计算
        
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
    if DEBUG:
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

    if DEBUG:
        cv_show('Grouped Contours with Manual Centers', grouped_image, 0)

    for i, group in enumerate(digit_groups):
        color = colors[i % len(colors)]
        for props in group:
            x, y, w, h = props['x'], props['y'], props['w'], props['h']
            cv.rectangle(grouped_image, (x, y), (x+w, y+h), color, 1)

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
                # cv.rectangle(grouped_image, (min_x, min_y), (max_x, max_y), color, 1)
            else:
                min_x = min(props['x'] for props in group)
                max_x = max(props['x'] + props['w'] for props in group)
                min_y = min(props['y'] for props in group)
                max_y = max(props['y'] + props['h'] for props in group)
                cv.rectangle(grouped_image, (min_x, min_y), (max_x, max_y), color, 1)
                # cv.putText(grouped_image, f"Group {i}", (min_x, min_y-5), 
                #         cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            print(f"Group {i} 边界框: x={min_x}, y={min_y}, w={max_x - min_x}, h={max_y - min_y}")

    if DEBUG:
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
        if DEBUG:
            cv_show(f'3. Digit {i} Region', digit_resized, 5)
        
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
            
            # 创建彩色图像用于重合显示
            overlap_img = np.zeros((digit_resized_resized.shape[0], digit_resized_resized.shape[1], 3), dtype=np.uint8)
            # 将原始二值图像放在红色通道
            overlap_img[:, :, 2] = digit_resized_resized  # 红色通道
        
            # 将模板放在绿色通道
            overlap_img[:, :, 1] = template  # 绿色通道
            # combined_img = np.hstack((digit_resized_resized, template))
            
            if DEBUG:
                cv_show(f'3. Digit {i} Region', overlap_img, 5)

            # 更新最佳匹配
            if score > best_score:
                best_score = score
                best_match = digit 
        
        recognized_digits.append(best_match)
        confidence_scores.append(best_score)
        
        # 在图像上绘制结果
        # cv.rectangle(matched_image, (min_x, min_y), (max_x, max_y), colors[i], 2)
        cv.putText(matched_image, f"{best_match}", 
                  (min_x, min_y+10), cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
        cv.putText(grouped_image, f"{best_match}", 
                  (min_x, min_y+10), cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
        
        print(f"数字 {i+1}: 识别为 {best_match}, 置信度: {best_score:.2f}")
    
    if DEBUG:
        cv_show('4. Final Matching Result', matched_image, 0)
    if DEBUG:
        cv_show('5. Final grouped Result', grouped_image, 0)
    return recognized_digits, grouped_image, digit_regions, confidence_scores
    # return recognized_digits, matched_image, digit_regions


""" To do list: debug this function """ 
def improved_template_matching_with_dot(roi_binary, digits_dict, manual_centers=None, manual_dot_centers=None,
                                    size_number=None, size_dot=None):
    """
    针对段码数字的改进模板匹配方法
    
    参数:
        roi_binary: 预处理后的二值图像（白字黑底）
        digits_dict: 数字模板字典
        manual_centers: 手动提供的数字中心点列表
        size_number: 每个数字的固定大小 (宽, 高)，如果为 None 则根据轮廓计算
        size_dot: 小数点的固定大小 (宽, 高)
        
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
    if DEBUG:
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
        print("使用手动提供的中心点对数字进行分组")
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
    color_dot = (0, 255, 255)  # 小数点颜色

    # 绘制手动中心点
    for i, (cx, cy) in enumerate(manual_centers):
        cv.circle(grouped_image, (int(cx), int(cy)), 1, colors[i], -1)
        # cv.putText(grouped_image, f"Center {i}", (int(cx)+5, int(cy)), 
        #           cv.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1)
    for i, (cx, cy) in enumerate(manual_dot_centers):
        cv.circle(grouped_image, (int(cx), int(cy)), 1, color_dot, -1)

    if DEBUG:
        cv_show('1 Grouped Contours with Manual Centers', grouped_image, 0)

    for i, group in enumerate(digit_groups):
        color = colors[i % len(colors)]
        for props in group:
            x, y, w, h = props['x'], props['y'], props['w'], props['h']
            cv.rectangle(grouped_image, (x, y), (x+w, y+h), color, 1)
            
        # cv_show('2. Grouped Contours with Manual Centers', grouped_image, 0)

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
                # cv.rectangle(grouped_image, (min_x, min_y), (max_x, max_y), color, 1)
            else:
                min_x = min(props['x'] for props in group)
                max_x = max(props['x'] + props['w'] for props in group)
                min_y = min(props['y'] for props in group)
                max_y = max(props['y'] + props['h'] for props in group)
                cv.rectangle(grouped_image, (min_x, min_y), (max_x, max_y), color, 1)
                # cv.putText(grouped_image, f"Group {i}", (min_x, min_y-5), 
                #         cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            print(f"Group {i} 边界框: x={min_x}, y={min_y}, w={max_x - min_x}, h={max_y - min_y}")

    if DEBUG:
        cv_show('2. Grouped Contours with Manual Centers', grouped_image, 0)


    # for i, group in enumerate(dot_groups):
    #     color = color_dot
    #     for props in group:
    #         x, y, w, h = props['x'], props['y'], props['w'], props['h']
    #         cv.rectangle(grouped_image, (x, y), (x+w, y+h), color, 1)
            
    #     # cv_show('2. Grouped Contours with Manual Centers', grouped_image, 0)

    #     # 计算组的边界框
    #     if group:
    #         if size_dot is not None:
    #             # pass
    #             cx, cy = manual_dot_centers[i]
    #             w_num, h_num = int(size_dot[0]), int(size_dot[1])
    #             min_x = int(cx - w_num / 2)
    #             min_y = int(cy - h_num / 2)
    #             max_x = min_x + w_num
    #             max_y = min_y + h_num
    #             # 限制在图像范围内
    #             min_x = max(0, min_x)
    #             min_y = max(0, min_y)
    #             max_x = min(grouped_image.shape[1], max_x)
    #             max_y = min(grouped_image.shape[0], max_y)
    #             # cv.rectangle(grouped_image, (min_x, min_y), (max_x, max_y), color, 1)
    #         else:
    #             min_x = min(props['x'] for props in group)
    #             max_x = max(props['x'] + props['w'] for props in group)
    #             min_y = min(props['y'] for props in group)
    #             max_y = max(props['y'] + props['h'] for props in group)
    #             cv.rectangle(grouped_image, (min_x, min_y), (max_x, max_y), color, 1)
    #             # cv.putText(grouped_image, f"Group {i}", (min_x, min_y-5), 
    #             #         cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    #         print(f"Group {i} 边界框: x={min_x}, y={min_y}, w={max_x - min_x}, h={max_y - min_y}")
    
    # if DEBUG:
    #     cv_show('2. Grouped Contours with Manual Centers', grouped_image, 0)

    

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
        if DEBUG:
            cv_show(f'3. Digit {i} Region', digit_resized, 1)
        
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
        # cv.rectangle(matched_image, (min_x, min_y), (max_x, max_y), colors[i], 2)
        cv.putText(matched_image, f"{best_match}", 
                  (min_x, min_y+10), cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
        cv.putText(grouped_image, f"{best_match}", 
                  (min_x, min_y+10), cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
        
        print(f"数字 {i+1}: 识别为 {best_match}, 置信度: {best_score:.2f}")
    
    if DEBUG:
        cv_show('4. Final Matching Result', matched_image, 0)

    return recognized_digits, grouped_image, digit_regions


def find_dot(roi_binary, manual_dot_centers=None, size_dot=None):
    
    """
    在二值化图像中查找小数点位置
    
    参数:
        roi_binary: 预处理后的二值图像（白字黑底）
        manual_dot_centers: 手动提供的小数点中心点列表
        size_dot: 小数点的固定大小 (宽, 高)"""
    
    if manual_dot_centers is None or size_dot is None:
        return []
    
    roi_binary_test = roi_binary.copy()
    roi_binary_test = cv.cvtColor(roi_binary_test, cv.COLOR_GRAY2BGR)
    
    dot_flags = []
    w_dot, h_dot = int(size_dot[0]), int(size_dot[1])
    for (cx, cy) in manual_dot_centers:
        min_x = int(cx - w_dot / 2)
        min_y = int(cy - h_dot / 2)
        max_x = min_x + w_dot
        max_y = min_y + h_dot
        # 限制在图像范围内
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(roi_binary.shape[1], max_x)
        max_y = min(roi_binary.shape[0], max_y)
        roi = roi_binary[min_y:max_y, min_x:max_x]
        # 判断区域内是否有白色像素（即有小数点）
        has_dot = np.count_nonzero(roi) > 0
        dot_flags.append(has_dot)

        if DEBUG:
            
            cv.rectangle(roi_binary_test, (min_x, min_y), (max_x, max_y), (255, 255, 0), 2)
            cv_show("Dot ROI", roi_binary_test, 0)

    return dot_flags


if __name__ == "__main__":
    img = cv.imread("template/Segment_digital_tube_number_with_dot.png")
    digit_groups, digits_dict = prepocess_template(img)
    
    if 1:
        recorder_image = cv.imread("images/test_1024_frame_002090.jpg")
        # recorder_image = cv.imread("output_frames/frame_000000.jpg")
        box = [633, 225, 75, 37]
        angle = -4
        threshold = 150
    else:
        recorder_image = cv.imread("output_frames/frame_005120.jpg")

        box = [780, 145, 250, 125]
        angle = -1.5
        threshold = 110
        
    
    roi_binary = preprocess_target(recorder_image, bbox=box, angle=angle, threshold=threshold)


    print("\n尝试改进的模板匹配...")
    manual_centers = [(11, 25), (36, 25), (61, 25), (86, 25)]   # 长宽已插值为100x50
    # 长宽已插值为100x50, size_number=(23,40)
    roiSize_of_digital_number = (20, 38)
    recognized_digits_improved, matched_image_improved, digit_regions, _ = improved_template_matching(
        roi_binary, digits_dict, manual_centers=manual_centers, size_number=roiSize_of_digital_number)
    

    manual_dot_centers = [(22, 40), (47, 40)] 
    # 长宽已插值为100x50, size_number=(23,40), size_dot=(2,2)
    roiSize_of_dot = (2, 2)

    dot_flags = find_dot(roi_binary, manual_dot_centers=manual_dot_centers, size_dot=roiSize_of_dot)
    print(f"dot_flags={dot_flags}")


    # recognized_digits_improved, matched_image, digit_regions = improved_template_matching_with_dot(
    #     roi_binary, digits_dict, manual_centers=manual_centers, manual_dot_centers=manual_dot_centers,
    #       size_number=roiSize_of_digital_number, size_dot=roiSize_of_dot)
