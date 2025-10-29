# @Author: Zheng Pan
# @Emial: pan_zheng@nwpu.edu.cn
# Copyright (c) 2025 Zheng Pan
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cv2 as cv
import os


def video_to_frames(video_path, output_folder, prefix='frame', start_time=0, end_time=None, frame_interval=10):
    """
    将视频转换为序列图片，支持指定时间范围

    参数:
        video_path (str): 视频文件路径
        output_folder (str): 输出图片的文件夹路径
        prefix (str): 输出图片的前缀名
        start_time (float): 开始时间秒,默认为0
        end_time (float): 结束时间(秒),默认为None(视频结尾）
        frame_interval (int): 帧间隔,默认为10(每10帧保存一张图片)
    """
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 读取视频文件
    cap = cv.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return

    # 获取视频属性[1,2](@ref)
    fps = cap.get(cv.CAP_PROP_FPS)  # 帧率
    print("视频的帧率 (FPS):", fps)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))  # 总帧数
    duration = total_frames / fps  # 视频总时长（秒）

    print(f"视频信息: 帧率={fps:.2f}, 总帧数={total_frames}, 时长={duration:.2f}秒")

    # 计算开始和结束帧[1,5](@ref)
    start_frame = int(start_time * fps)
    if end_time is None:
        end_frame = total_frames - 1
    else:
        end_frame = min(int(end_time * fps), total_frames - 1)

    print(f"提取范围: 第{start_frame}帧到第{end_frame}帧 (时间: {start_time}秒到{end_time if end_time else duration}秒)")

    # 设置起始位置[2,5](@ref)
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = start_frame
    saved_count = 0
    # frames_to_process = 0

    while frame_count <= end_frame:
        # 逐帧读取视频
        ret, frame = cap.read()

        if not ret:
            break
        if frame_count % frame_interval == 0:
            # 构造输出图片的文件名和路径
            frame_filename = f"{prefix}_{frame_count:06d}.jpg"
            output_path = os.path.join(output_folder, frame_filename)

            # 保存帧为图片文件
            success = cv.imwrite(output_path, frame)

            if success:
                saved_count += 1
                if saved_count % 100 == 0:  # 每保存100帧打印一次进度
                    progress = ((frame_count - start_frame) / (end_frame - start_frame)) * 100
                    print(f"进度: {progress:.1f}% - 已保存 {saved_count} 帧...")
            else:
                print(f"保存失败: {output_path}")

        frame_count += 1

    cap.release()
    print(f"处理完成！成功保存 {saved_count} 张图片到 {output_folder}")


# 使用示例
if __name__ == "__main__":
    if 1:
        video_path = "test_record/turbojet_test_2025-1024-1413.mp4"  # 输入视频
    else:
        video_path = "test_record/my_data/turbojet_test_2025-1028-2110.mp4"  # 输入视频
        
    output_folder = "output_frames"  # 输出文件夹名称
    
    # 示例1: 提取整个视频
    # video_to_frames(video_path, output_folder)
    
    # 示例2: 提取从第0秒到第3秒的帧
    video_to_frames(video_path, output_folder, start_time=120, end_time=180,frame_interval=10)
