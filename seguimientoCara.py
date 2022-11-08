import math
import cv2
import mediapipe as mp
import time 

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
index_list =                    [(33, 7), (7, 163), (163, 144), (144, 145),
                                (145, 153), (153, 154), (154, 155), (155, 133),
                                (33, 246), (246, 161), (161, 160), (160, 159),
                                (159, 158), (158, 157), (157, 173), (173, 133),
                                (263, 249), (249, 390), (390, 373), (373, 374),
                               (374, 380), (380, 381), (381, 382), (382, 362),
                               (263, 466), (466, 388), (388, 387), (387, 386),
                               (386, 385), (385, 384), (384, 398), (398, 362),
                               (263, 249), (249, 390), (390, 373), (373, 374),
                               (374, 380), (380, 381), (381, 382), (382, 362),
                               (263, 466), (466, 388), (388, 387), (387, 386),
                               (386, 385), (385, 384), (384, 398), (398, 362)]

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
                mp_drawing.draw_landmarks(frame,face_landmarks,
                   index_list,
                   mp_drawing.DrawingSpec(color = (0 ,255, 0), thickness = 1, circle_radius = 1),
                   mp_drawing.DrawingSpec(color = (255 ,0 ,255), thickness = 1))

        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()


