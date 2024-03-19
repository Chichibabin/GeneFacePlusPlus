import cv2
import numpy as np
import torch
from tqdm import tqdm
from deep_3drecon.deep_3drecon_models.bfm import ParametricFaceModel
from scipy.io import loadmat
import pyvista as pv

# 加载3DMM系数
coeff_path = '/root/autodl-tmp/shijieqi/GeneFacePlusPlus/data/processed/videos/XueLi/coeff_fit_mp.npy'
coeff_dict = np.load(coeff_path, allow_pickle=True).item()

# 初始化3DMM模型
face_model = ParametricFaceModel(bfm_folder='deep_3drecon/BFM', camera_distance=10, focal=1015, keypoint_mode='mediapipe')
face_model.to(torch.device("cuda:0"))
bfm_folder = 'deep_3drecon/BFM'

# 加载BFM模型的三角形信息
bfm_model_path = f'{bfm_folder}/BFM_model_front.mat'
bfm_data = loadmat(bfm_model_path)
triangles = bfm_data['tri'] - 1  # 减1以适配Python的0索引

# 将三角形面转换为pyvista兼容的格式
faces_pyvista = np.hstack(np.concatenate(([[3]] * len(triangles), triangles), axis=1)).astype(np.int32)

# 提取第一帧的系数
id_para = torch.FloatTensor(coeff_dict['id'][0:1]).cuda()
exp_para = torch.FloatTensor(coeff_dict['exp'][0:1]).cuda()
euler_angle = torch.FloatTensor(coeff_dict['euler'][0:1]).cuda()
trans = torch.FloatTensor(coeff_dict['trans'][0:1]).cuda()

# # 重建3D人脸网格
# vertices = face_model.compute_face_vertex(id_para, exp_para, euler_angle, trans)
# vertices = vertices.detach().cpu().numpy()[0]


# # 创建一个pyvista的PolyData对象
# pv.start_xvfb()
# mesh_pyvista = pv.PolyData(vertices, faces=faces_pyvista)
# mesh_pyvista.plot()

# # 保存网格为PLY文件
# mesh_pyvista.save('3d_face_mesh_pyvista.ply')

# 设置视频参数
frame_width = 1536
frame_height = 512
fps = 25
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/root/autodl-tmp/shijieqi/GeneFacePlusPlus/data/raw/videos/XueLi_3d_faces_video.mp4', fourcc, fps, (frame_width, frame_height))

pv.start_xvfb()

# 读取原视频
video_path = '/root/autodl-tmp/shijieqi/GeneFacePlusPlus/data/raw/videos/XueLi.mp4'
cap = cv2.VideoCapture(video_path)

# 渲染前100帧
for frame_idx in tqdm(range(len(coeff_dict['exp'])), desc='Rendering 3D Models'):
    # 读取原视频帧
    ret, original_frame = cap.read()
    if not ret:
        break
    # 调整原视频帧大小
    original_frame = cv2.resize(original_frame, (512, 512))
    # 提取当前帧的系数
    id_para = torch.FloatTensor(coeff_dict['id'][frame_idx:frame_idx + 1]).cuda()
    exp_para = torch.FloatTensor(coeff_dict['exp'][frame_idx:frame_idx + 1]).cuda()
    euler_angle = torch.FloatTensor(coeff_dict['euler'][frame_idx:frame_idx + 1]).cuda()
    trans = torch.FloatTensor(coeff_dict['trans'][frame_idx:frame_idx + 1]).cuda()

    # 重建3D人脸网格
    vertices = face_model.compute_face_vertex(id_para, exp_para, euler_angle, trans)
    vertices = vertices.detach().cpu().numpy()[0]

    # 创建一个pyvista的PolyData对象
    mesh_pyvista = pv.PolyData(vertices, faces=faces_pyvista)

    # 使用pyvista渲染3D网格并将其转换为NumPy数组
    plotter = pv.Plotter(off_screen=True)
    plotter.window_size = [512, 512]
    plotter.add_mesh(mesh_pyvista, color='white')
    plotter.set_background('black')
    
    plotter.view_zy()
    plotter.show(auto_close=False)
    img_3d_yz = plotter.screenshot()

    # 调整3D渲染图像大小
    img_3d_yz = cv2.resize(img_3d_yz, (512,512))
    
    plotter.view_xy(negative=True)
    plotter.show(auto_close=False)
    img_3d_xy = plotter.screenshot()

    # 调整3D渲染图像大小
    img_3d_xy = cv2.resize(img_3d_xy, (512,512))

    plotter.close()

    # 将原视频帧和3D渲染帧拼接
    combined_frame = np.hstack((original_frame,img_3d_xy, img_3d_yz))

    # 写入输出视频
    out.write(combined_frame)

# 释放资源
cap.release()
out.release()