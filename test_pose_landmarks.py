from gluoncv import model_zoo
import cv2
from models.yolov5 import Y5Detect
from utils.pose_landmarks import get_pose
from utils.utils import draw_boxes_yolo


detector_yolo = Y5Detect(weights='Body_Card/yolov5s.pt')
pose_net = model_zoo.get_model('simple_pose_resnet50_v1b', pretrained=True)
wanted_joints = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                 'left_hip', 'right_hip']

# Note that we can reset the classes of the detector to only include
# human, so that the NMS process is faster.

image_path = 'samples/test.jpg'
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

#detect person
bboxes_yolo, labels_yolo, scores_yolo = detector_yolo.predict(image_rgb)
bbox_yolo = bboxes_yolo[0]
image_bgr_person = image_bgr[bbox_yolo[1]:bbox_yolo[3], bbox_yolo[0]:bbox_yolo[2], :]

#get pose
image_bgr_person, pred_coords, confidence = get_pose(image_bgr_person, pose_net, wanted_joints)

#visualize
image_bgr[bbox_yolo[1]:bbox_yolo[3], bbox_yolo[0]:bbox_yolo[2], :] = image_bgr_person
draw_boxes_yolo(image_bgr, bboxes_yolo, scores_yolo, labels_yolo)

cv2.imshow('Img', image_bgr)
cv2.waitKey()