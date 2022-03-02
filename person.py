from models.yolov5 import Y5Detect
from utils.utils import draw_boxes_yolo, get_person, draw_bboxes, draw_poly
from utils.detect_card import detect_card
from insightface.model_zoo.retinaface import RetinaFace
from insightface.model_zoo.landmark import Landmark
from gluoncv import model_zoo
from utils.mouth_landmark import get_mouth, mouth_open, draw_mouth
from utils.pose_landmarks import get_pose
from utils.utils_pose import normalize_pose_landmarks, calc_feature_pose, butter_lowpass_filter
import cv2
import numpy as np

class Person_tracking:
    def __init__(self, number_frame_track=32, number_sequence_frame=10):
        # box [x1 y1 x2 y2]
        self.frame_track = number_frame_track
        self.number_sequence_frame = number_sequence_frame
        self.motion_threshold = 2

        self.Card_model = detect_card(weights_person='Body_Card/yolov5s.pt',
                                 weights_card='Body_Card/model_detect_card.pt',
                                 weights_classification='Body_Card/classifier.h5',
                                 use_cuda=False)
        self.Face_model = RetinaFace(model_file='Face_Landmarks/det_10g.onnx')
        self.Landmark_model = Landmark(model_file="Face_Landmarks/2d106det.onnx")
        self.Pose_net = model_zoo.get_model('mobile_pose_mobilenetv3_large', pretrained=True)

        self.wanted_joints = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist',
                         'right_wrist',
                         'left_hip', 'right_hip']

        # Status
        self.count_frames = 0
        self.number_tour_guide_card = 0
        self.total_card_detected = 0

        self.mouth_features = []
        self.pose_features = []

        self.is_speaking = False
        self.is_presenting = False
        self.speaking_time = 0
        self.presenting_time = 0
    def forward(self, person_image):
        self.count_frames += 1

        #face detection
        faces, _ = self.Face_model.detect(person_image, max_num=0, metric='default', input_size=(640, 640))
        face_box, landmark = [], []
        if len(faces) != 0:
            face_box = faces.astype(np.int)[0]
            #landmarks detection
            landmark = get_mouth(person_image, face_box, self.Landmark_model)
            mouth_open_percent = mouth_open(landmark)
            self.mouth_features.append(mouth_open_percent)
            #self.forward_speaking(mouth_open_percent)

        # card detection
        bboxes_card, labels_card, scores_card, scores_classification = self.Card_model.predict(person_image, show=False)

        # pose detection
        person_image_out, pred_coords, confidence = get_pose(person_image, self.Pose_net, self.wanted_joints)
        landmarks = pred_coords[0].asnumpy()
        landmarks = normalize_pose_landmarks(landmarks)
        features = calc_feature_pose(landmarks)
        self.pose_features.append(features)
        #self.forward_presenting(pred_coords)

        # visualize
        person_image_out = draw_mouth(person_image_out, face_box, landmark)
        person_image_out = draw_bboxes(person_image_out, bboxes_card, labels_card, scores_card, scores_classification)
        h, w = person_image.shape[:2]
        person_image_out = cv2.putText(person_image_out,
                                   str(self.count_frames),
                                   (10, h-10),
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   1, (255, 0,  0), 2, cv2.LINE_AA)
        #cv2.imshow('person', person_image)

        if self.count_frames == self.frame_track:
            #Speaking recognition
            standard_deviation = np.std(self.mouth_features)
            mean = np.mean(self.mouth_features)
            if standard_deviation > 5 and mean > 10:
                self.speaking_time += 1
                self.is_speaking = True
            else:
                self.is_speaking = False
            self.mouth_features = []
            print("Is Speaking: ", self.is_speaking)

            #Presenting recognition
            self.pose_features = np.array(self.pose_features)
            standard_deviation = []
            for i in range(0, 4):
                feature = self.pose_features[:, i]
                feature = butter_lowpass_filter(feature)
                standard_deviation.append(np.std(feature))
            out = np.mean(standard_deviation)
            if out>self.motion_threshold:
                self.is_presenting = True
                self.presenting_time += 1
            else:
                self.is_presenting = False
            self.pose_features = []
            print("Is Presenting: ", self.is_presenting)
            print('-'*20)
            self.count_frames = 0
        return person_image_out









