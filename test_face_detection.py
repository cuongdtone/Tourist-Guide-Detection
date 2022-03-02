from models.yolov5 import Y5Detect
from utils.utils import draw_boxes_yolo, get_person, draw_bboxes, draw_poly
from utils.detect_card import detect_card
from insightface.model_zoo.retinaface import RetinaFace
from insightface.model_zoo.landmark import Landmark
from gluoncv import model_zoo
from utils.mouth_landmark import get_mouth, mouth_open, draw_mouth
from utils.pose_landmarks import get_pose
import cv2
import numpy as np

face_model = RetinaFace(model_file='Face_Landmarks/det_10g.onnx')

input_path = 0
cap = cv2.VideoCapture(input_path)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if ret:
            #face detection
        faces, _ = face_model.detect(frame, max_num=0, metric='default', input_size=(640, 640))

        face_box = faces.astype(np.int)[0]
        print(face_box)
        cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (0, 0, 255), 1)
        cv2.imshow("Video", frame)
        cv2.waitKey(5)
video.release()


