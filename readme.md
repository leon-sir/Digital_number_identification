
# Number Identification
## Overview
This project is used for turbojet thrusts reader identification

(本项目参考@深蓝的回音，信用卡识别项目，https://bbs.huaweicloud.com/blogs/247279)

## Setup the environment

```bash
conda create -n env_cv python=3.10

# my conda env name
conda activate env_cv
```

```bash
pip install opencv-python --upgrade 
pip install numpy imutils matplotlib
```


## 使用说明
Firstly, find and record RoI of each segement number on the template:
```bash
python3 scripts/find_roi.py
```

<div style="text-align: center;">
  <img src="images/read_me_image1.png" alt="find roi" width="300">
</div>

Then check if the number is extracted correctly. (轮廓中心在RoI就行)

<div style="text-align: center;">
  <img src="images/read_me_image2.png" alt="extracted_number" width="800">
</div>

