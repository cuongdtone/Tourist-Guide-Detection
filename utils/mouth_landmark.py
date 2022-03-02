import math
import cv2
import numpy as np
from insightface.model_zoo.landmark import Landmark
from insightface.app.common import Face
def draw_poly(image,
              pts,
              is_closed=False,
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

def get_height(top, bottom):
    sum = 0
    for i in [0,1,2]:
        distance = math.sqrt((top[i][0] - bottom[i][0]) ** 2 +
                             (top[i][1] - bottom[i][1]) ** 2)
        sum += distance
    return sum / 3
def mouth_open(landmark):
    top_lip = [[landmark[63], landmark[71], landmark[67]],
               [landmark[66], landmark[62], landmark[70]]]
    top_lip_height = get_height(top_lip[0], top_lip[1])
    bottom_lip = [[landmark[54], landmark[60], landmark[57]],
               [landmark[56], landmark[53], landmark[59]]]
    bottom_lip_height = get_height(bottom_lip[0], bottom_lip[1])
    mouth = [[landmark[66], landmark[62], landmark[70]],
               [landmark[54], landmark[60], landmark[57]]]
    mouth_height = get_height(mouth[0], mouth[1])
    return mouth_height/(top_lip_height+bottom_lip_height+mouth_height) * 100
def get_mouth(img, face_box, model_face_landmark):
    face_box_class = {'bbox': face_box}
    face_box_class = Face(face_box_class)
    landmark = model_face_landmark.get(img, face_box_class)
    return landmark
def draw_mouth(img, face_box, landmark):
    if len(face_box)!=0:
        face = img[face_box[1]:face_box[3], face_box[0]:face_box[2]:]
        #print(face.shape)
        #landmark = faces[0].landmark_2d_106.astype(np.int)
        for point in range(len(landmark)):
            landmark[point] = landmark[point] - np.asarray([face_box[0], face_box[1]])  # get position face
            # new shape
            #landmark[point][0] = landmark[point][0] * 500 / (box[2] - box[0])
            #landmark[point][1] = landmark[point][1] * 500 / (box[3] - box[1])
        # img[2,2,:] = [255, 0, 0]
        l1 = [landmark[52], landmark[55], landmark[56], landmark[53], landmark[59], landmark[58], landmark[61]]
        l2 = [landmark[65], landmark[54], landmark[60], landmark[57], landmark[69]]
        l3 = [landmark[65], landmark[66], landmark[62], landmark[70], landmark[69]]
        l4 = [landmark[52], landmark[64], landmark[63], landmark[71], landmark[67], landmark[68], landmark[61]]
        # img = draw_poly(img, landmark[62:71])

        #img = np.zeros([500, 500, 3], np.uint8)
        face_mask = np.zeros(face.shape, np.uint8)
        face_mask = draw_poly(face_mask, l4, color_bgr=[0, 255, 0])
        face_mask = draw_poly(face_mask, l1, color_bgr=[0, 255, 0])

        face_mask = draw_poly(face_mask, l2, color_bgr=[0, 0, 255])
        face_mask = draw_poly(face_mask, l3, color_bgr=[0, 0, 255])

        #face = cv2.resize(face, (500,500))
        #face = change_brightness(face, -30)
        face = cv2.add(face, face_mask)
        img[face_box[1]:face_box[3], face_box[0]:face_box[2]:] = face
        cv2.rectangle(img, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (0,0,255), 1)
        #head = img[box[1]-25:box[3]+25, box[0]-25:box[2]+25:]
    return img