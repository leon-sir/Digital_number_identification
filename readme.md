
# Number Identification
## Overview
This project is used for turbojet thrusts reader identification

(本项目参考@深蓝的回音，信用卡识别项目，https://bbs.huaweicloud.com/blogs/247279)

## Setup the environment

```bash
conda create -n env_cv python=3.10

conda activate env_cv
```

```bash
pip install opencv-python --upgrade 
pip install numpy imutils matplotlib
```

```bash
# Add the project root to PYTHONPATH
pip install -e .
```

## 使用说明
Firstly, find and record RoI of each segement number on the template:
```bash
python3 scripts/find_roi.py --img template/Segment_digital_tube_number_with_dot.png
```

<div style="text-align: center;">
  <img src="images/read_me_image1.png" alt="find roi" width="300">
</div>

Then check if the number is extracted correctly. (轮廓中心在RoI就行)

<div style="text-align: center;">
  <img src="images/read_me_image2.png" alt="extracted_number" width="800">
</div>

Find RoI in target images
```bash
python3 scripts/find_roi.py --img output_frames/frame_002090.jpg
```

## 识别
```bash
python3 number_identification.py
```