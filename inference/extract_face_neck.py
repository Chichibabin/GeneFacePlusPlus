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

def extract_face_skin(frame, segmenter, frame_idx, size):
    segmap = segmenter._cal_seg_map(frame, return_onehot_mask=False)
    mask = np.zeros_like(segmap, dtype=np.uint8)
    mask[segmap == 2] = 1  # body_skin
    mask[segmap == 3] = 1  # face_skin
    alpha_channel = np.zeros_like(segmap, dtype=np.uint8)
    alpha_channel[(mask == 1)] = 255  # Set alpha to 255 for body_skin and face_skin pixels
    neck_long = 20

    
    # Create a face mesh object
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True)

    # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.
    results = mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # dilate iterations
    dilite_it = 3
    
    # blur sigma
    blur_sigma = 5

    # Draw face landmarks of each face.
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get the coordinates of all face landmarks
            landmark_coords = np.array([(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in face_landmarks.landmark])
            # Find the highest landmark
            # midle_landmark = landmark_coords[midle_point_index][1]
            # left_eye_point = landmark_coords[468]      
            # righr_eye_point = landmark_coords[473]
            # Set the part above the highest landmark to transparent
            # alpha_channel[:midle_landmark] = 0
            
            # Find the lowest landmark
            lowest_landmark = np.max(landmark_coords[:, 1])
            # Set the part below the lowest landmark + 20 pixels to transparent
            alpha_channel[lowest_landmark + neck_long:] = 0
            # cv2.circle(frame, (landmark_coords[midle_point_index][0], midle_landmark), radius=5, color=(0, 0, 255), thickness=-1)
            # cv2.circle(frame,(left_eye_point[0], left_eye_point[1]), radius=5, color=(0, 0, 255), thickness=-1)
            # cv2.circle(frame,(righr_eye_point[0], righr_eye_point[1]), radius=5, color=(0, 0, 255), thickness=-1) 
            
    
    # Dilate the alpha channel before applying Gaussian blur
    kernel = np.ones((5,5),np.uint8)
    alpha_channel = cv2.dilate(alpha_channel, kernel, iterations = dilite_it)
    # alpha_channel = cv2.erode(alpha_channel, kernel, iterations = 0)

    # Blur the alpha channel to smooth the edges
    alpha_channel = cv2.GaussianBlur(alpha_channel, (15, 15), blur_sigma)
    
    result_frame = np.dstack((frame, alpha_channel))  # Add the alpha channel to the frame
    # 根据scle调整result_frame的大小
    result_frame = cv2.resize(result_frame, (int(size), int(size)))
    cv2.imwrite(f'tmp_frames/frame_{frame_idx:04d}.png', result_frame)
    mp_face_mesh.close()

def process_video(input_video_path, segmenter, raw_video_path=None):
    cap_raw = cv2.VideoCapture(raw_video_path)
    ret, frame = cap_raw.read()
    if not ret:
        print("Can't read the first frame.")
        exit(1)
    mp_face_detection = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    results = mp_face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
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
    cap_raw.release()

    cap = cv2.VideoCapture(input_video_path)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        extract_face_skin(frame, segmenter, frame_idx, size=crop_size)
        frame_idx += 1

    cap.release()
    return frame_idx-1
    
def get_video_duration(video_path):
    cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {video_path}"
    duration = subprocess.check_output(cmd, shell=True).decode().strip()
    return float(duration)

if __name__ == '__main__':
    seg_model = MediapipeSegmenter()
    video_name = "ZhengXinXin"
    input_video_path = f'./{video_name}_fake.mp4'
    real_person_video_path = f'./{video_name}_test/{video_name}_raw_25.mp4'
    frame_pattern = f'tmp_frames/frame_%04d.png'
    output_video_path = f'./{video_name}_test/{video_name}_no_sound.mp4'
    final_output_video_path = f'./{video_name}_test/{video_name}_fake.mp4'
    
    # Create a directory to store the frames
    os.makedirs(f'tmp_frames', exist_ok=True)
    os.makedirs(f'{video_name}_test', exist_ok=True)
    frame_count = process_video(input_video_path, seg_model, real_person_video_path)
    # frame_count = 307

    # 初始化 MediaPipe
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode = False, min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks = True)
    mp_halfface_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode = False, min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks = True)

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
            half_face_result =mp_halfface_mesh.process(half_face_rgb)
            with open(f'./data/raw/videos/{video_name}_coordinates.txt', 'r') as file:
                line = file.readline()
                numbers = line.split(',')
                first_two_numbers = numbers[:2]
                offset_x, offset_y = [int(num) for num in first_two_numbers]
            
            # 找出 half_face 中透明通道不为 0 的像素
            alpha_channel = half_face[:, :, 3]
            non_zero_alpha_pixels = np.nonzero(alpha_channel > 0)
            # 按原视频计算偏移量
            y_indices = non_zero_alpha_pixels[0] 
            x_indices = non_zero_alpha_pixels[1] 
                                    
            # 如果检测到人脸
            # if results.multi_face_landmarks and half_face_result.multi_face_landmarks:
            #     # 获取第一个人脸的关键点
            #     face_landmarks = results.multi_face_landmarks[0]
            #     half_face_landmarks = half_face_result.multi_face_landmarks[0]                            
            #     # 提取左右眼瞳孔中心的坐标作为基准
            #     left_eye_center = (int(face_landmarks.landmark[468].x * frame.shape[1]), int(face_landmarks.landmark[468].y * frame.shape[0]))
            #     right_eye_center = (int(face_landmarks.landmark[473].x * frame.shape[1]), int(face_landmarks.landmark[473].y * frame.shape[0]))

            #     left_eye_center_half_face = (int(half_face_landmarks.landmark[468].x * half_face.shape[1]), int(half_face_landmarks.landmark[468].y * half_face.shape[0]))
            #     right_eye_center_half_face = (int(half_face_landmarks.landmark[473].x * half_face.shape[1]), int(half_face_landmarks.landmark[473].y * half_face.shape[0]))

            #     # 计算两眼中心点的平均坐标作为基准点
            #     base_point = ((left_eye_center[0] + right_eye_center[0]) // 2, (left_eye_center[1] + right_eye_center[1]) // 2)
            #     target_point = ((left_eye_center_half_face[0] + right_eye_center_half_face[0]) // 2, (left_eye_center_half_face[1] + right_eye_center_half_face[1]) // 2)

            #     # 计算偏移量
            #     y_offset = base_point[1] - target_point[1]
            #     x_offset = base_point[0] - target_point[0]
                
            #     # 将偏移量写入 CSV 文件
            #     writer.writerow([i, x_offset, y_offset])

            #     # 按面部关键点坐标计算偏移量
            #     y_indices = non_zero_alpha_pixels[0] + y_offset
            #     x_indices = non_zero_alpha_pixels[1] + x_offset

            # 检查坐标是否在 frame 的边界内
            valid_indices = (y_indices >= 0) & (y_indices < frame.shape[0]) & (x_indices >= 0) & (x_indices < frame.shape[1])  
            # 提取有效的像素坐标
            valid_y_indices = y_indices[valid_indices] + offset_y
            valid_x_indices = x_indices[valid_indices] + offset_x
            
            # 计算 Alpha 值
            alpha = alpha_channel[non_zero_alpha_pixels[0][valid_indices], non_zero_alpha_pixels[1][valid_indices]] / 255.0

            # 应用 Alpha Blending
            frame[valid_y_indices, valid_x_indices] = (alpha[:, None] * half_face_rgb[non_zero_alpha_pixels[0][valid_indices], non_zero_alpha_pixels[1][valid_indices]]) + ((1 - alpha[:, None]) * frame[valid_y_indices, valid_x_indices])
                
            # 将处理后的帧写入输出视频
            out.write(frame)

    # 释放资源
    cap.release()
    out.release()
    mp_face_mesh.close()
    mp_halfface_mesh.close()
    cv2.destroyAllWindows()
    cmd = f"ffmpeg -i {output_video_path} -i {input_video_path} -c copy -map 0:0 -map 1:1 -shortest {final_output_video_path}"
    os.system(cmd)
    
    
    # ffmpeg -i /root/autodl-tmp/shijieqi/GeneFacePlusPlus/ZhengXinXin_test/ZhengXinXin_raw.mp4 -vf fps=25,scale=w=720:h=1280 -qmin 1 -q:v 1 /root/autodl-tmp/shijieqi/GeneFacePlusPlus/ZhengXinXin_test/ZhengXinXin_raw_25.mp4
