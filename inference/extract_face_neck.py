import csv
import os
import cv2
import numpy as np
import subprocess
from data_gen.utils.mp_feature_extractors.mp_segmenter import MediapipeSegmenter
import mediapipe as mp
from tqdm import tqdm
from scipy.ndimage import binary_erosion, binary_dilation

midle_point_index = 5
left_point_index = 205
right_point_index = 425

def line_side(point1, point2, point):
    # Calculate the coefficients of the line equation ax + by + c = 0
    a = point2[1] - point1[1]
    b = point1[0] - point2[0]
    c = point2[0] * point1[1] - point1[0] * point2[1]

    # Calculate the value of the line equation for the given point
    value = a * point[0] + b * point[1] + c

    # Determine the side of the line the point is on
    if value > 1:
        return True
    else:
        return False
    

def extract_face_skin(frame, segmenter, frame_idx):
    segmap = segmenter._cal_seg_map(frame, return_onehot_mask=False)
    mask = np.zeros_like(segmap, dtype=np.uint8)
    mask[segmap == 2] = 1  # body_skin
    mask[segmap == 3] = 1  # face_skin
    alpha_channel = np.zeros_like(segmap, dtype=np.uint8)
    alpha_channel[(mask == 1)] = 255  # Set alpha to 255 for body_skin and face_skin pixels

    
    # # Create a face mesh object
    # mp_face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.
    # results = mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # # Draw face landmarks of each face.
    # if results.multi_face_landmarks:
    #     for face_landmarks in results.multi_face_landmarks:
    #         # Get the coordinates of all face landmarks
    #         landmark_coords = np.array([(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in face_landmarks.landmark])

    #         # Find the highest landmark
    #         midle_landmark = landmark_coords[midle_point_index][1]
                    

    #         # Set the part above the highest landmark to transparent
    #         alpha_channel[:midle_landmark] = 0
    
    # Dilate the alpha channel before applying Gaussian blur
    kernel = np.ones((5,5),np.uint8)
    alpha_channel = cv2.dilate(alpha_channel, kernel, iterations = 10)

    # # Blur the alpha channel to smooth the edges
    # alpha_channel = cv2.GaussianBlur(alpha_channel, (15, 15), 0)

    result_frame = np.dstack((frame, alpha_channel))  # Add the alpha channel to the frame
    cv2.imwrite(f'tmp_frames/frame_{frame_idx:04d}.png', result_frame)

# 用于预处理原视频，进行把脖子拉长的效果。在只希望只贴回人脸不贴回颈部的情况下使用。
def pre_process_video(input_video_path, output_video_path,segmenter):
    cap = cv2.VideoCapture(input_video_path)

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 创建 tqdm 对象
    pbar = tqdm(total=total_frames, desc="Processing video")

    # 创建面部网格对象
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 将 BGR 图像转换为 RGB，并使用 MediaPipe Face Mesh 处理它。
        results = mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 获取颈部像素掩码
        segmap = segmenter._cal_seg_map(frame, return_onehot_mask=False)
        neck_part = np.zeros_like(segmap, dtype=np.uint8)
        neck_part[segmap == 2] = 1
        

        # 假设颈部像素掩码为 neck_part，L为向上拉长的像素数
        L = 20

        # 找到颈部的顶端
        top_neck = np.argmax(neck_part, axis=0)

        # 从颈部的顶端开始向上拉长L个像素
        for col in range(frame.shape[1]):
            if top_neck[col] - L >= 0:
                frame[top_neck[col] - L:top_neck[col], col] = frame[top_neck[col], col]

        # 将帧写入输出视频
        out.write(frame)
                
        # 更新进度条
        pbar.update(1)

    # 关闭进度条
    pbar.close()

    # 释放 VideoWriter 和 VideoCapture 对象
    out.release()
    cap.release()

def process_video(input_video_path, segmenter):
    cap = cv2.VideoCapture(input_video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        extract_face_skin(frame, segmenter, frame_idx)
        frame_idx += 1

    cap.release()
    return frame_idx-1
    
def get_video_duration(video_path):
    cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {video_path}"
    duration = subprocess.check_output(cmd, shell=True).decode().strip()
    return float(duration)

if __name__ == '__main__':
    seg_model = MediapipeSegmenter()
    input_video_path = '/root/autodl-tmp/shijieqi/GeneFacePlusPlus/xueli_for_com.mp4'
    real_person_video_path = '/root/autodl-tmp/shijieqi/GeneFacePlusPlus/data/raw/videos/XueLi_short.mp4'
    frame_pattern = 'tmp_frames/frame_%04d.png'
    frame_pattern_with_ld = 'tmp_frames_with_ld/frame_%04d.png'
    output_video_path = 'xueli_com_demo_full_face.mp4'
    output_video_path_with_ld = 'xueli_com_demo_with_ld.mp4'
    tmp_output_video_path = '/root/autodl-tmp/shijieqi/GeneFacePlusPlus/xueli_for_com_noface.mp4'
    
    # pre_process_video(real_person_video_path, tmp_output_video_path, seg_model)
    
    # Create a directory to store the frames
    os.makedirs('tmp_frames', exist_ok=True)
    os.makedirs('tmp_frames_with_ld', exist_ok=True)
    frame_count = process_video(input_video_path, seg_model)
    # frame_count = 307

    # 初始化 MediaPipe
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

    # 打开视频文件
    cap = cv2.VideoCapture(real_person_video_path)
    fps = 25

    # 初始化视频编写器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    # 创建一个 CSV 文件
    with open('tmp_offsets.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # 写入标题行
        writer.writerow(["frame", "x_offset", "y_offset"])
        # 遍历视频的每一帧
        for i in tqdm(range(frame_count), desc="Processing video frames"):
            ret, frame = cap.read()
            if not ret:
                break

            # 在这里使用 MediaPipe 获取人脸关键点信息
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_face_mesh.process(image)
            
            # 加载对应的图片序列中的半张人脸
            image_file = frame_pattern % (i + 1)
            half_face = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
            half_face_rgb = half_face[:, :, :3]
            half_face_result = mp_face_mesh.process(half_face_rgb)
            
            # 如果检测到人脸
            if results.multi_face_landmarks and half_face_result.multi_face_landmarks:
                # 获取第一个人脸的关键点
                face_landmarks = results.multi_face_landmarks[0]
                half_face_landmarks = half_face_result.multi_face_landmarks[0]
                # 提取5号关键点的坐标作为基准
                base_point = (int(face_landmarks.landmark[midle_point_index].x * frame.shape[1]), int(face_landmarks.landmark[midle_point_index].y * frame.shape[0]))
                target_point = (int(half_face_landmarks.landmark[midle_point_index].x * frame.shape[1]), int(half_face_landmarks.landmark[midle_point_index].y * frame.shape[0]))                
                
                # # 在 frame 上画红点
                # cv2.circle(frame, base_point, radius=5, color=(0, 0, 255), thickness=-1)
                # # 在 half_face 上画绿点
                # cv2.circle(half_face, target_point, radius=5, color=(0, 255, 0), thickness=-1)
                
                # 将半张人脸贴回视频帧中
                y_offset = base_point[1] - target_point[1]
                x_offset = base_point[0] - target_point[0]
                
                # 将偏移量写入 CSV 文件
                writer.writerow([i, x_offset, y_offset])
                
                # 找出 half_face 中透明通道不为 0 的像素
                alpha_channel = half_face[:, :, 3]
                non_zero_alpha_pixels = np.nonzero(alpha_channel > 0)

                # 计算这些像素在 frame 中的坐标
                y_indices, x_indices = non_zero_alpha_pixels
                y_indices += y_offset
                x_indices += x_offset

                # 检查坐标是否在 frame 的边界内
                valid_indices = (y_indices < frame.shape[0]) & (x_indices < frame.shape[1])

                # 将 half_face 中透明通道不为 0 的像素的 RGB 值赋值给 frame 中对应位置的像素
                frame[y_indices[valid_indices], x_indices[valid_indices]] = half_face_rgb[non_zero_alpha_pixels[0][valid_indices], non_zero_alpha_pixels[1][valid_indices]]
                
                # 找出 half_face 中透明通道不为 0 的像素
                alpha_channel = half_face[:, :, 3]
                non_zero_alpha_pixels = np.nonzero(alpha_channel > 0)
                
                # 使用 Canny 边缘检测找出 half_face 中的边缘
                edges = cv2.Canny(alpha_channel, 100, 200)
                # 创建一个指定宽度的 mask
                kernel_size = 5  # 调整这个值来改变 mask 的宽度
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                mask = cv2.dilate(edges, kernel, iterations=1)
                # 计算这些像素在 frame 中的坐标
                y_indices, x_indices = np.nonzero(mask > 0)
                y_indices += y_offset
                x_indices += x_offset

                # 检查坐标是否在 frame 的边界内
                valid_indices = (y_indices < frame.shape[0]) & (x_indices < frame.shape[1])

                # # 将 mask 对应的像素在 frame 中画成红色
                # frame[y_indices[valid_indices], x_indices[valid_indices]] = [0, 0, 255]
                
                # 创建一个和 frame 一样大小的 mask
                frame_mask = np.zeros_like(frame[:,:,0])
                frame_mask[y_indices[valid_indices], x_indices[valid_indices]] = 255
                # 对 mask 应用高斯模糊
                frame_mask = cv2.GaussianBlur(frame_mask, (kernel_size, kernel_size), 0)    

                # 使用 cv2.inpaint 函数修复 frame 中的 mask 对应的像素
                # frame = cv2.inpaint(frame, frame_mask, 2, cv2.INPAINT_TELEA)
                
            # 将处理后的帧写入输出视频
            out.write(frame)

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()