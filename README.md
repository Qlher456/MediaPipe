# MediaPipe

这是利用谷歌公司的MediaPipe库


MediaPipe 是一款由 Google Research 开发并开源的多媒体机器学习模型应用框架。
MediaPipe Face Landmarker 任务允许检测图像和视频。可以使用此任务来识别人类的面部表情，应用面部滤镜和效果，并创建虚拟形象。该任务输出 3D 人脸标志。

MediaPipe人脸关键点检测模型包含了478个3D关键点


关键代码：
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=5,      # Maximum number of detected faces
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)


max_num_faces：要检测的最大人脸数
refine_landmarks：是否进一步细化眼睛和嘴唇周围的地标坐标，并输出虹膜周围的其他地标。
min_detection_confidence：人脸检测的置信度
min_tracking_confidence：人脸跟踪的置信度

![Cut-output](https://github.com/user-attachments/assets/5d022790-3824-48c9-9e5f-241a860a5d39)

![Image-input](https://github.com/user-attachments/assets/84c13c29-382d-4ee5-9312-02c75c1a3837)

![Image-output](https://github.com/user-attachments/assets/bc7b0a01-82ad-4511-83aa-687ffeb3340c)


其中MediaPipe.py是基础代码：使用MediaPipe库调用摄像头直接转换3D人脸
Cut.py是将转换的3D人脸保存下来
Image.py是使用MediaPipe库检测图像
