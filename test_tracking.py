from models.yolov5 import Y5Detect
from utils.utils import draw_boxes_yolo,draw_bboxes_track, get_person, draw_poly, xyxy_conf_clss, xyxys_to_xywhs
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from utils.torch_utils import select_device

import cv2
import numpy as np
import torch

device = 'cpu'
device = select_device(device)

person_model = Y5Detect(weights="Body_Card/yolov5s.pt", use_cuda=False)
input_path = 'samples/Tram_Trai_30.mp4'


cap = cv2.VideoCapture(input_path)
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video', 1280, 720)

# deepsort
cfg = get_config()
cfg.merge_from_file('deep_sort/configs/deep_sort.yaml')
deep_sort_model = 'osnet_x0_25'
deepsort = DeepSort(deep_sort_model,
                    device,
                    max_dist=cfg.DEEPSORT.MAX_DIST,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    )

while True:
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes, labels, scores = person_model.predict(frame_rgb)
        bboxes, labels, scores = get_person(bboxes, labels, scores, labels_idx=True)

        #pred = xyxy_conf_clss(bboxes, labels, scores)
        #pred = [torch.Tensor(pred)]
        bbox_xywh = torch.Tensor(xyxys_to_xywhs(bboxes))
        confidences = torch.Tensor(scores)
        labels = torch.Tensor(labels)
        # print(bbox_xywh)
        # print(confidences)
        # print(labels)
        outputs = deepsort.update(bbox_xywh, confidences, labels, frame)
        frame = draw_bboxes_track(frame, outputs)
        cv2.imshow('Video', frame)
        cv2.waitKey(5)
