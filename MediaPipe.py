# 利用摄像头检测人脸

import mediapipe as mp
import numpy as np
import cv2


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=10,      # Maximum number of detected faces
                                  refine_landmarks=True,    # Whether to further refine the landmark coordinates around the eyes and lips
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

while True:

    ret, img = cap.read()
    height, width, channels = np.shape(img)
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(img_RGB)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw a facial mesh
            mp_drawing.draw_landmarks(image=img,
                                      landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            # Draw facial contours
            mp_drawing.draw_landmarks(image=img,
                                      landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            # Draw iris contours
            mp_drawing.draw_landmarks(image=img,
                                      landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_IRISES,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
            # Draw facial keypoints
            # if face_landmarks:
            #     for i in range(478):
            #         pos_x = int(face_landmarks.landmark[i].x * width)
            #         pos_y = int(face_landmarks.landmark[i].y * height)
            #         cv2.circle(img, (pos_x, pos_y), 3, (0, 255, 0), -1)

    num_faces = len(results.multi_face_landmarks)
    print(f"Detected {num_faces} faces")

    cv2.imshow('faces', img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()

