import cv2
import numpy as np

def draw_landmarks_on_video(video_path, landmarks_path, output_video_path):
    # 加载特征点
    landmarks = np.load(landmarks_path)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频的帧率和尺寸信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 绘制特征点
        for lm in landmarks[frame_count]:
            cv2.circle(frame, (int(lm[0]), int(lm[1])), 2, (0, 255, 0), -1)

        # 写入带有特征点的帧
        out.write(frame)

        frame_count += 1
        if frame_count >= len(landmarks):
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# 使用示例
video_path = '/root/autodl-tmp/shijieqi/GeneFacePlusPlus/data/raw/videos/XueLi.mp4'
landmarks_path = '/root/autodl-tmp/shijieqi/GeneFacePlusPlus/data/processed/videos/XueLi/lms_2d.npy'
output_video_path = '/root/autodl-tmp/shijieqi/GeneFacePlusPlus/data/raw/videos/XueLi_lm2d.mp4'
draw_landmarks_on_video(video_path, landmarks_path, output_video_path)
