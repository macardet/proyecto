import math
import cv2
import mediapipe as mp
import time 

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)


with mp_face_mesh.FaceMesh(
    static_image_mode = False,
    max_num_faces = 1,
    min_detection_confidence = 0.5) as face_mesh:

    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.flip(frame,1)
        frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                for index in index_list:
                    x = int(face_landmarks.landmark[index.x]*width)
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()

