# @Author: Zheng Pan
# @Emial: pan_zheng@nwpu.edu.cn
# Copyright (c) 2025 Zheng Pan
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from scripts import video_to_frames, cv_show, sort_contours, point_in_rect, my_resize  # noqa: F401
import cv2 as cv
import numpy as np


# RoI: [x,y,w,h], (x, y): center of the number in temlpate/segment_digital_tube_number_with_dot.png
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


def prepocess_template(img):
    # 新增：动态调整并保持宽高比的 resize + pad 函数
    def resize_and_pad_template(bin_img, target_size):
        """
        将二值模板调整到 target_size (height, width)，保持纵横比并用黑(0)填充
        """
        th, tw = target_size
        h, w = bin_img.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((th, tw), dtype=np.uint8)
        scale = min(tw / float(w), th / float(h))
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv.resize(bin_img, (new_w, new_h), interpolation=cv.INTER_NEAREST)
        pad_w = tw - new_w
        pad_h = th - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        padded = cv.copyMakeBorder(resized, top, bottom, left, right, cv.BORDER_CONSTANT, value=0)
        return padded
    
    """
    处理七位段码数字,模板从1-9,0
    """
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
        shrink_margin = 0.0     # 可调整，越小保留越紧，试试 0.05 ~ 0.20

        if group:
            pts = np.vstack([cnt.reshape(-1, 2) for cnt in group])
            min_x_abs, min_y_abs = pts.min(axis=0).astype(int)
            max_x_abs, max_y_abs = pts.max(axis=0).astype(int)

            # 基于手动bbox尺寸决定 padding（避免切掉笔画）
            pad_px = max(1, int(shrink_margin * max(w, h)))

            crop_min_x = max(0, min_x_abs - pad_px)
            crop_min_y = max(0, min_y_abs - pad_px)
            crop_max_x = min(img.shape[1], max_x_abs + pad_px)
            crop_max_y = min(img.shape[0], max_y_abs + pad_px)

            crop_w = crop_max_x - crop_min_x
            crop_h = crop_max_y - crop_min_y

            # 创建模板并把轮廓绘制到模板坐标系
            digit_template = np.zeros((crop_h, crop_w), dtype=np.uint8)
            for cnt in group:
                offset_cnt = cnt - [crop_min_x, crop_min_y]
                cv.drawContours(digit_template, [offset_cnt], -1, 255, -1)  # 填充轮廓

            # # 创建一个与边界框相同大小的空白图像
            # digit_template = np.zeros((int(h), int(w)), dtype=np.uint8)

            # # 将轮廓绘制到模板上
            # for cnt in group:
            #     # 调整轮廓坐标到模板坐标系
            #     offset_cnt = cnt - [int(x - w/2), int(y - h/2)]
            #     cv.drawContours(digit_template, [offset_cnt], -1, 255, -1)  # 填充轮廓

            # # 根据是否为小数点调整模板大小
            # if i == 10:  # 小数点使用原始尺寸
            #     digit_template = cv.resize(digit_template, (32, 32))
            # else:  # 数字使用标准尺寸
            #     digit_template = cv.resize(digit_template, (57, 88))
            if i == 10:  # 小数点使用 32x32
                digit_template = resize_and_pad_template(digit_template, (32, 32))
            else:  # 数字使用标准尺寸 高x宽 = 88x57
                digit_template = resize_and_pad_template(digit_template, (88, 57))

            # 存储模板，对小数点使用特殊键值'.'
            if i < 9:
                digits_dict[i + 1] = digit_template
            elif i == 9:
                digits_dict[0] = digit_template
            else:
                digits_dict['.'] = digit_template

    # 显示所有生成的模板（按 1-9, 0, 小数点 顺序）
    display_order = list(range(1, 10)) + [0, '.']
    for k in display_order:
        if k in digits_dict:
            label = 'Dot' if k == '.' else f'Digit {k} Template'
            cv_show(label, digits_dict[k], 0)


    cv_show('4. Grouped Contours with Manual BBoxes', img_with_groups, 0)     # 显示分组结果

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


def process_target(image, **kwargs):
    """
    处理目标图像，支持多种参数组合

    参数:
        image: 输入图像
        **kwargs: 可选参数，包括:
            - bbox: 边界框 [x, y, w, h]
            - angle: 旋转角度
            - threshold: 二值化阈值
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

    rotated_full_image = cv.warpAffine(image, M, image.shape[:2])
    # cv_show('Rotated Full Image', rotated_full_image)

    roi_image = rotated_full_image[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
    cv_show('ROI After Rotation', roi_image)

    roi_gray = cv.cvtColor(roi_image, cv.COLOR_BGR2GRAY)
    cv_show('gray', roi_gray)

    """ 边缘提取, threshold由屏幕亮度决定,背景色亮的需要使用 cv.THRESH_BINARY_INV """
    threshold = params['threshold']
    _, roi_binary = cv.threshold(roi_gray, threshold, 255, cv.THRESH_BINARY_INV)
    cv_show('gray', roi_binary)

    return roi_binary


def improved_template_matching(roi_binary, digits_dict, manual_centers=None):
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
        cv.circle(grouped_image, (int(cx), int(cy)), 3, colors[i], -1)
        cv.putText(grouped_image, f"Center {i}", (int(cx)+5, int(cy)), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1)

    for i, group in enumerate(digit_groups):
        color = colors[i % len(colors)]
        for props in group:
            x, y, w, h = props['x'], props['y'], props['w'], props['h']
            cv.rectangle(grouped_image, (x, y), (x+w, y+h), color, 1)

        # 计算组的边界框
        if group:
            min_x = min(props['x'] for props in group)
            max_x = max(props['x'] + props['w'] for props in group)
            min_y = min(props['y'] for props in group)
            max_y = max(props['y'] + props['h'] for props in group)
            cv.rectangle(grouped_image, (min_x, min_y), (max_x, max_y), color, 2)
            cv.putText(grouped_image, f"Group {i}", (min_x, min_y-5), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

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


def main():

    """ .mp4 2 .png """
    # video_path = "test_record/1437485690.mp4"
    # output_folder = "output_frames"  # 输出文件夹名称
    # video_to_frames(video_path, output_folder, start_time=30, end_time=60)

    """ extract number contour from template  """
    img = cv.imread("template/Segment_digital_tube_number_with_dot.png")
    # cv_show('template', img, 2)

    # digit_groups(list)储存数字和轮廓, digits_dict(dict)存储数字(key))和对应的模板图像(value)
    digit_groups, digits_dict = prepocess_template(img)
    

    print("extract number contour from template successfully")

    # 初始化卷积核
    rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 3))
    sqKernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

    # 待分析图片读入,预处理
    recorder_image = cv.imread("output_frames/frame_001340.jpg")
    # roi_binary = process_target(recorder_image, bbox=[262, 215, 86, 37], angle=-9, threshold=140)
    roi_binary = process_target(recorder_image, bbox=[258, 245, 86, 37], angle=-15, threshold=140)

    # return
    # tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, rectKernel)     # tophat可以突出图片中明亮的区域，过滤掉较暗的部分
    # cv_show('tophat', tophat)

    """template matching"""
    # # 基本模板匹配
    # recognized_digits, matched_image = template_matching(roi_binary, digits_dict)
    # cv_show('Template Matching Result', matched_image, 0)
    # print(recognized_digits)

    # 改进的模板匹配
    print("\n尝试改进的模板匹配...")
    manual_centers = [(9, 19), (31, 19), (53, 19), (74, 19)]

    recognized_digits_improved, matched_image_improved, digit_regions = improved_template_matching(
        roi_binary, digits_dict, manual_centers=manual_centers)

    # 显示每个数字的区域
    for i, region_info in enumerate(digit_regions):
        cv_show(f'Digit {i+1} Region', region_info['resized'], 500)

    # recognized_digits_improved, matched_image_improved = improved_template_matching(roi_binary, digits_dict)
    cv_show('Improved Template Matching Result', matched_image_improved, 0)
    print(recognized_digits_improved)


    print("done")


if __name__ == "__main__":
    main()
