from utils.utils import *
from models.mobilenet import mobilenetv3
from models.yolov5 import Y5Detect
import os

class detect_card:
    def __init__(self, weights_person='', weights_card='', weights_classification='', use_cuda=False):

        self.y5_card = Y5Detect(weights=weights_card, use_cuda=use_cuda)
        self.model_classification = mobilenetv3(weights=weights_classification, use_cuda=use_cuda)
        self.corner_threshold = 0
    def predict(self, image_bgr, show=True):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        bboxes, labels, scores = self.y5_card.predict(image_rgb)

        #segmentation
        self.bboxes_card = []
        self.labels_card = []
        self.scores_card = []
        self.scores_classification = []
        for ind, k in enumerate(labels):
            if k == 'Card':
                self.scores_card.append(scores[ind])
                topleft = []
                topright = []
                bottomright = []
                bottomleft = []
                for ind2, k2 in enumerate(labels):  # find 4 corner for card
                    if k2 == 'topleft':
                        _, pivot = cal_iou(bboxes[ind2], bboxes[ind])
                        if pivot > self.corner_threshold:
                            topleft = center(bboxes[ind2])
                    if k2 == 'topright':
                        _, pivot = cal_iou(bboxes[ind2], bboxes[ind])
                        if pivot > self.corner_threshold:
                            topright = center(bboxes[ind2])
                    if k2 == 'bottomleft':
                        _, pivot = cal_iou(bboxes[ind2], bboxes[ind])
                        if pivot > self.corner_threshold:
                            bottomleft = center(bboxes[ind2])
                    if k2 == 'bottomright':
                        _, pivot = cal_iou(bboxes[ind2], bboxes[ind])
                        if pivot > self.corner_threshold:
                            bottomright = center(bboxes[ind2])
                if topleft != [] and topright != [] and bottomright != [] and bottomleft != []:
                    self.bboxes_card.append([topleft, topright, bottomright, bottomleft])
                    card = crop_card(image_bgr, [topleft, topright, bottomright, bottomleft])
                    pred, score = self.model_classification.predict(card)
                    self.labels_card.append(pred)  # the co the phan lop
                    self.scores_classification.append(score)

                else:
                    x1, y1, x2, y2 = bboxes[ind]
                    self.bboxes_card.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                    self.labels_card.append('Khong xac dinh')  # the khong the phan lop
                    self.scores_classification.append(0)
        if show:
            self.visualization(image_bgr)
        return self.bboxes_card, self.labels_card, self.scores_card, self.scores_classification
    def visualization(self, image):
        image_show = draw_bboxes(image,
                                 self.bboxes_card,
                                 self.labels_card,
                                 self.scores_card,
                                 self.scores_classification)
        cv2.imshow('detections', image_show)
        cv2.waitKey()