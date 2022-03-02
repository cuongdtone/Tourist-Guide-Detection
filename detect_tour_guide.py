from person import Person_tracking
from utils.utils import get_person, draw_boxes_yolo
from models.yolov5 import Y5Detect
import cv2

person_model = Y5Detect(weights="Body_Card/yolov5s.pt", use_cuda=False)
person_track = Person_tracking()

input_path = '/home/cuong/Desktop/Speaking/TedTalks/video1_clip1.mp4'
cap = cv2.VideoCapture(input_path)
while True:
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes, labels, scores = person_model.predict(frame_rgb)
        bboxes, labels, scores = get_person(bboxes, labels, scores, threshold=0.8)

        # only one person in frame
        if len(bboxes)!=0:
            person_box = bboxes[0]
            person_image = frame[person_box[1]:person_box[3], person_box[0]:person_box[2]]

            person_image = person_track.forward(person_image)

            frame[person_box[1]:person_box[3], person_box[0]:person_box[2]] = person_image
            frame = draw_boxes_yolo(frame, bboxes, scores, labels, person_model.class_names)

        cv2.imshow("Video", cv2.resize(frame, (1280, 720)))
        key = cv2.waitKey(5)
        if key == ord('q'):
            break
    else:
        break