import cv2
import numpy as np
import mediapipe as mp
import os

video_name = "ZhengXinXin"

# 初始化 mediapipe 的人脸检测模型
mp_face_detection = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 打开视频
cap = cv2.VideoCapture(f'./data/raw/videos/{video_name}_raw.mp4')

# 获取视频的宽度和高度及帧率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'width: {width}, height: {height}, fps: {fps}, frame_count: {frame_count}')

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

    # 获取最顶端关键点的坐标（额头中央）
    top_point = face_landmarks.landmark[10]
    top_point_coords = (top_point.x, top_point.y)

    # 获取最底端关键点的坐标（下巴最低点）
    bottom_point = face_landmarks.landmark[152]
    bottom_point_coords = (bottom_point.x, bottom_point.y)

    print("Top point coordinates:", top_point_coords)
    print("Bottom point coordinates:", bottom_point_coords)

    # 计算面部最顶端到最底端的长度
    face_length = np.sqrt((top_point_coords[0] - bottom_point_coords[0]) ** 2 + (top_point_coords[1] - bottom_point_coords[1]) ** 2) * frame.shape[1]
    print("Face long axis length:", face_length)

    # 根据面部长度和目标分辨率动态计算裁剪区域的大小
    target_face_ratio = 1/5
    crop_size = int(face_length / target_face_ratio)

    # 计算裁剪区域的左上角和右下角的坐标
    x1 = max(0, x - crop_size // 2)
    y1 = max(0, y - crop_size // 2)
    x2 = min(width, x + crop_size // 2)
    y2 = min(height, y + crop_size // 2)

    # 保存裁切窗口的四个角的坐标到一个 txt 文件中
    with open(f'./data/raw/videos/{video_name}_coordinates.txt', 'w') as f:
        f.write(f'{crop_size},{x1},{y1},{x2},{y2}\n')

    cmd = f'ffmpeg -i ./data/raw/videos/{video_name}_raw.mp4 -vf "crop={crop_size}:{crop_size}:{x1}:{y1},scale=512:512" ./data/raw/videos/{video_name}.mp4'
    os.system(cmd)
