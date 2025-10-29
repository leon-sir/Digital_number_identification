# @Author: Zheng Pan
# @Emial: pan_zheng@nwpu.edu.cn
# Copyright (c) 2025 Zheng Pan
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import shutil


def clear_output_folder():
    current_dir = "output_frames"    # 获取储存目录

    if not os.path.exists(current_dir):
            print(f"文件夹 {current_dir} 不存在，无需清空")
            return
    
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


if __name__ == "__main__":
    clear_output_folder()