# 利用摄像头检测人脸，绘制3D图像

import mediapipe as mp
import numpy as np
import cv2

# 初始化 MediaPipe 人脸网格模型
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=10,  # 检测的最大人脸数
                                  refine_landmarks=True,  # 是否进一步细化眼睛和嘴唇周围的地标
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# 初始化绘图工具
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 启动摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头帧
    ret, img = cap.read()
    if not ret:
        print("无法读取摄像头帧")
        break

    height, width, channels = np.shape(img)
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 使用 FaceMesh 处理图像
    results = face_mesh.process(img_RGB)

    # 如果检测到人脸，绘制面部网格并保存图像
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 绘制面部网格
            mp_drawing.draw_landmarks(image=img,
                                      landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            # 绘制面部轮廓
            mp_drawing.draw_landmarks(image=img,
                                      landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            # 绘制虹膜轮廓
            mp_drawing.draw_landmarks(image=img,
                                      landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_IRISES,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

        # 显示检测到的人脸数量
        num_faces = len(results.multi_face_landmarks)
        print(f"Detected {num_faces} faces")

        # 保存处理好的图像
        output_filename = "Cut-output.jpg"
        cv2.imwrite(output_filename, img)
        print(f"处理后的 3D 人脸图像已保存为 {output_filename}")

        # 退出循环，结束程序
        break

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()
