import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import os

video_name = "ZhengXinXin"


# 初始化 mediapipe 的人脸检测模型
mp_face_detection = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 打开视频
cap = cv2.VideoCapture(f'/root/autodl-tmp/shijieqi/GeneFacePlusPlus/data/raw/videos/{video_name}_raw.mp4')

# 获取视频的宽度和高度及帧率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


# 创建一个 VideoWriter 对象来输出裁切后的视频
out = cv2.VideoWriter(f'/root/autodl-tmp/shijieqi/GeneFacePlusPlus/data/raw/videos/{video_name}_no_sound.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (512, 512))

# 创建一个 VideoWriter 对象来输出裁切示意视频
out_demo = cv2.VideoWriter(f'/root/autodl-tmp/shijieqi/GeneFacePlusPlus/data/raw/videos/{video_name}_for_cut.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

# 读取第一帧
ret, frame = cap.read()
if not ret:
    print("Can't read the first frame.")
    exit(1)

# 使用 mediapipe 进行人脸检测
results = mp_face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# 获取第一个检测到的人脸的 5 号关键点的坐标
if results.multi_face_landmarks:
    face_landmarks = results.multi_face_landmarks[0]
    x = int(face_landmarks.landmark[5].x * frame.shape[1])
    y = int(face_landmarks.landmark[5].y * frame.shape[0])

    # 计算裁切区域的左上角和右下角的坐标
    x1 = max(0, x - 256)
    y1 = max(0, y - 256)
    x2 = min(width, x + 256)
    y2 = min(height, y + 256)

    # 保存裁切窗口的四个角的坐标到一个 txt 文件中
    with open(f'/root/autodl-tmp/shijieqi/GeneFacePlusPlus/data/raw/videos/{video_name}_coordinates.txt', 'w') as f:
        f.write(f'{x1},{y1},{x2},{y2}\n')

# 创建一个进度条
pbar = tqdm(total=frame_count)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 在原视频上绘制裁切窗口的绿色边框
    demo_frame = cv2.rectangle(frame.copy(), (x1, y1), (x2, y2), (0, 255, 0), 2)
    demo_frame = cv2.circle(demo_frame, (x, y), 2, (0, 255, 0), 2)

    # 裁切帧
    cropped_frame = frame[y1:y2, x1:x2]

    # 将裁切后的帧调整为 512x512 大小
    resized_frame = cv2.resize(cropped_frame, (512, 512))

    # 将帧写入输出视频
    out.write(resized_frame)

    # 将带有裁切窗口的帧写入裁切示意视频
    out_demo.write(demo_frame)
    
    # 更新进度条
    pbar.update(1)

# 关闭进度条
pbar.close()

# 释放资源
cap.release()
out.release()
out_demo.release()
# 将音频从原视频复制到裁切后的视频
os.system(f'ffmpeg -i /root/autodl-tmp/shijieqi/GeneFacePlusPlus/data/raw/videos/{video_name}_raw.mp4 -i /root/autodl-tmp/shijieqi/GeneFacePlusPlus/data/raw/videos/{video_name}_no_sound.mp4 -c copy -map 0:a:0 -map 1:v:0 -shortest /root/autodl-tmp/shijieqi/GeneFacePlusPlus/data/raw/videos/{video_name}.mp4')

# 将音频从原视频复制到裁切示意视频
os.system(f'ffmpeg -i /root/autodl-tmp/shijieqi/GeneFacePlusPlus/data/raw/videos/{video_name}_raw.mp4 -i /root/autodl-tmp/shijieqi/GeneFacePlusPlus/data/raw/videos/{video_name}_for_cut.mp4 -c copy -map 0:a:0 -map 1:v:0 -shortest /root/autodl-tmp/shijieqi/GeneFacePlusPlus/data/raw/videos/{video_name}_for_cut_with_sound.mp4')
# ffmpeg -i /root/autodl-tmp/shijieqi/GeneFacePlusPlus/data/raw/videos/ZhengXinXin.mp4 -ss 00:00:02 -to $(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 /root/autodl-tmp/shijieqi/GeneFacePlusPlus/data/raw/videos/ZhengXinXin.mp4 | awk '{print int($1)-2}') -c copy /root/autodl-tmp/shijieqi/GeneFacePlusPlus/data/raw/videos/ZhengXinXin_raw.mp4