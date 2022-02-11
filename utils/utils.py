import numpy as np
import cv2

def cal_iou(true_box, pre_box):
    """
    function: calculator IOU value
    :param true_box: list bbox true
    :param pre_box: list bbox predict
    :return: IOU value is in the range 0-1
    """
    xmin_true_box, ymin_true_box, xmax_true_box, ymax_true_box = true_box
    xmin_pre_box, ymin_pre_box, xmax_pre_box, ymax_pre_box = pre_box
    if xmax_pre_box < xmin_true_box or ymax_pre_box < ymin_true_box:
        return 0.0, 0.0
    if xmin_pre_box > xmax_true_box or ymin_pre_box > ymax_true_box:
        return 0.0, 0.0
    True_box_area = (xmax_true_box - xmin_true_box + 1)*(ymax_true_box - ymin_true_box + 1)
    Pre_box_area = (xmax_pre_box - xmin_pre_box + 1)*(ymax_pre_box - ymin_pre_box + 1)
    xmin_inter = np.max([xmin_pre_box, xmin_true_box])
    ymin_inter = np.max([ymin_pre_box, ymin_true_box])
    xmax_inter = np.min([xmax_pre_box, xmax_true_box])
    ymax_inter = np.min([ymax_pre_box, ymax_true_box])
    Intersection_area = (xmax_inter - xmin_inter + 1)*(ymax_inter - ymin_inter + 1)
    Union_area = (True_box_area + Pre_box_area - Intersection_area)
    return Intersection_area/Union_area, True_box_area/Intersection_area

def center(bbox):
    x1, y1, x2, y2 = bbox
    return [int((x1+x2)/2), int((y1+y2)/2)]

def draw_poly(image,
              pts,
              is_closed=True,
              color_bgr=[255, 0, 0],
              size=0.01,  # 1%
              line_type=cv2.LINE_AA,
              is_copy=True):
    assert size > 0

    image = image.copy() if is_copy else image  # copy/clone a new image

    # calculate thickness

    h, w = image.shape[:2]
    if size > 0:
        short_edge = min(h, w)
        thickness = int(short_edge * size)
        thickness = 1 if thickness <= 0 else thickness
    else:
        thickness = -1

    # docs: https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#gaa3c25f9fb764b6bef791bf034f6e26f5
    cv2.polylines(img=image,
                  pts=[np.int32(pts)],
                  isClosed=is_closed,
                  color=color_bgr,
                  thickness=1,
                  lineType=line_type,
                  shift=0)
    return image
def four_point_transform(img, polygon):
    h, w, chanel = img.shape
    dst = np.array([(0, 0), (w, 0), (w, h), (0, h)], dtype="float32")
    pts = np.array(polygon, dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (w, h))
    #warped = change_brightness(warped, random.randint(-60, -20))
    #warped = cv2.blur(warped, ksize=(3, 3))
    return warped
def crop_card(image, polygon):
    polygon = np.array(polygon)
    #print(polygon)
    x = np.copy(polygon).T[0]
    y = np.copy(polygon).T[1]
    x.sort()
    y.sort()
    xmin = x[0]
    ymin = y[0]
    xmax = x[3]
    ymax = y[3]
    #print(xmin, xmax, ymin, ymax)
    card = image[ymin:ymax, xmin:xmax, :]
    x = polygon.T[0] - xmin
    y = polygon.T[1] - ymin
    new_polygon = (np.array([x, y])).T
    #print(x, y)
    card = four_point_transform(card, new_polygon)
    return card

def draw_bboxes(image, bboxes_card, labels_card, scores, scores_classification):
    image_show = image
    if len(labels_card)>0:
        for i in range(len(labels_card)):
            card = crop_card(image, bboxes_card[i]) #one card
            #image_name = i.split('/')[-1]
            h, w = card.shape[:2]
            #image_show = cv2.resize(image_show, (1280, 720))
            x = bboxes_card[i][2][0]
            y = bboxes_card[i][2][1]
            if False:
                image_show[y+25:h+y+25, x+25:w+x+25, :] = card
                cv2.arrowedLine(image_show, (x, y), (x+22, y+22), color=(0, 0, 255))
                cv2.rectangle(image_show, (x+25, y+25), (x+25+w, y+25+h), color=(0, 0, 255))
                ret, baseline = cv2.getTextSize(labels_card[i],
                                                cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1)
                cv2.rectangle(image_show, (x+25, y+25+h+ret[1]), (x+25 + ret[0], y+25+h), (0, 0, 255), -1)
                cv2.putText(image_show,
                            labels_card[i] + '-'+str(scores_classification[i]), (x+25, y+h+25+ret[1]-baseline),
                            fontFace=cv2.FONT_HERSHEY_PLAIN,
                            thickness=2,
                            fontScale=1,
                            color=(255,255,255)
                            )
            image_show = draw_poly(image_show, bboxes_card[i])
            cv2.putText(image_show,
                         labels_card[i]+'-%.2f'%(scores[i]), (bboxes_card[i][3][0], bboxes_card[i][3][1]),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        thickness=2,
                        fontScale=1,
                        color=(0,0,255))

    return image_show
def draw_boxes_yolo(image, boxes, scores=None, labels=None, class_names=None, line_thickness=2, font_scale=2.0,
               font_thickness=3):
    #num_classes = len(class_names)
    # colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]
    if scores is not None and labels is not None:
        for b, l, s in zip(boxes, labels, scores):
            if class_names is None:
                class_name = 'person'
                class_id = 0
            elif l not in class_names:
                class_id = int(l)
                class_name = class_names[class_id]
            else:
                class_name = l
                class_id = class_names.index(l)

            xmin, ymin, xmax, ymax = list(map(int, b))
            score = '{:.4f}'.format(s)
            # color = colors[class_id]
            color = (255, 0, 0)
            label = '-'.join([class_name, score])

            ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, line_thickness)
            cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), (255, 255, 255), -1)
            cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255),
                        font_thickness)
    elif labels is not None:
        for b, l in zip(boxes, labels):
            xmin, ymin, xmax, ymax = list(map(int, b))
            idx = class_names.index(l)
            color = colors[idx]

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    elif scores is not None:
        idx = 0
        for b, s in zip(boxes, scores):
            xmin, ymin, xmax, ymax = list(map(int, b))
            score = '{:.4f}'.format(s)
            color = colors[idx]
            label = '-'.join([score])
            idx += 1

            ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            # cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
            cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    else:
        color = (0, 255, 0)
        for b in boxes:
            xmin, ymin, xmax, ymax = list(map(int, b))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
    return image
def get_person(bboxes, labels, scores, threshold=0.5):
    new_bboxes, new_labels, new_scores = [], [], []
    for idx, label in enumerate(labels):
        if label =="person" and scores[idx]>threshold:
            new_bboxes.append(bboxes[idx])
            new_scores.append(scores[idx])
            new_labels.append(label)
    return new_bboxes, new_labels, new_scores

