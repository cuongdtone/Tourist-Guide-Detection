from matplotlib import pyplot as plt
from gluoncv import data
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
import cv2
import mxnet as mx
from utils.utils_pose import cv_plot_upper_keypoints, get_wanted_joints

# input image_person
# output: coords, visualize

def get_pose(image_person, pose_net, wanted_joints):
    h, w = image_person.shape[:2]
    old_dim = (w, h)
    image_rgb_person = cv2.cvtColor(image_person, cv2.COLOR_BGR2RGB)
    image_mx = mx.nd.array(image_rgb_person).astype('uint8')

    x, img = data.transforms.presets.ssd.transform_test(image_mx, short=512, max_size=350)
    class_IDs = mx.nd.array([[[0]]])
    scores = mx.nd.array([[[1.0]]])
    bounding_boxs = mx.nd.array([[[0, 0, 197, 350]]])

    pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs)
    predicted_heatmap = pose_net(pose_input)
    pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
    pred_coords, confidence = get_wanted_joints(pred_coords, confidence, wanted_joints)
    image_rgb_person = cv_plot_upper_keypoints(img, pred_coords, confidence, class_IDs, bounding_boxs, scores)
    image_rgb_person = cv2.resize(image_rgb_person, old_dim)
    image_rgb_person = cv2.cvtColor(image_rgb_person, cv2.COLOR_RGB2BGR)
    return image_rgb_person, pred_coords, confidence