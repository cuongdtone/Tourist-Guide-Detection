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

person_model = Y5Detect(weights="Body_Card/yolov5s.pt", use_cuda=False)
face_model = RetinaFace(model_file='Face_Landmarks/det_10g.onnx')
landmark_model = Landmark(model_file="Face_Landmarks/2d106det.onnx")
card_model = detect_card(weights_person='Body_Card/yolov5s.pt',
                    weights_card='Body_Card/model_detect_card.pt',
                    weights_classification='Body_Card/classifier.h5',
                    use_cuda=False)
pose_net = model_zoo.get_model('mobile_pose_mobilenetv3_large', pretrained=True)

wanted_joints = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                 'left_hip', 'right_hip']

input_path = '/home/cuong/Desktop/test case 3/test_3_10.mp4'
output_path = 'outputs/' + input_path.split('/')[-1]

cap = cv2.VideoCapture(input_path)
width = int(cap.get(3))
height = int(cap.get(4))
result = cv2.VideoWriter(output_path,
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (width, height))

while True:
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes, labels, scores = person_model.predict(frame_rgb)
        bboxes, labels, scores = get_person(bboxes, labels, scores)

        # only one person in frame
        person_box = bboxes[0]
        person_image = frame[person_box[1]:person_box[3], person_box[0]:person_box[2]]

            #face detection
        faces, _ = face_model.detect(person_image, max_num=0, metric='default', input_size=(640, 640))
        face_box = faces.astype(np.int)[0]

            #landmarks detection
        landmark = get_mouth(person_image, face_box, landmark_model)
        mouth_open_percent = mouth_open(landmark)
        print(mouth_open_percent)

            #card detection
        bboxes_card, labels_card, scores_card, scores_classification = card_model.predict(person_image, show=False)

            #pose detection
        person_image, pred_coords, confidence = get_pose(person_image, pose_net, wanted_joints)
        print(pred_coords)

        # visualize
        person_image = draw_mouth(person_image, face_box, landmark)
        person_image = draw_bboxes(person_image, bboxes_card, labels_card, scores_card, scores_classification)
        person_image = cv2.rectangle(person_image, (face_box[0], face_box[1]),
                                    (face_box[2], face_box[3]),
                                    (0, 0, 255), 2)
        frame[person_box[1]:person_box[3], person_box[0]:person_box[2]] = person_image
        #end detector

        frame = draw_boxes_yolo(frame, bboxes, scores, labels, person_model.class_names)

        result.write(frame)
        cv2.imshow("Video", cv2.resize(frame, (1280, 720)))
        cv2.waitKey(5)

video.release()
result.release()