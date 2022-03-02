from gluoncv.utils.filesystem import try_import_cv2
import mxnet as mx
import numpy as np
import cv2
import matplotlib.pyplot as plt

KEYPOINTS = {0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear', 5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow', 9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip', 13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle'}

def cv_plot_keypoints(img, coords, confidence, class_ids, bboxes, scores,
                      box_thresh=0.5, keypoint_thresh=0.2, scale=1.0, **kwargs):
    """Visualize keypoints with OpenCV.

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    coords : numpy.ndarray or mxnet.nd.NDArray
        Array with shape `Batch, N_Joints, 2`.
    confidence : numpy.ndarray or mxnet.nd.NDArray
        Array with shape `Batch, N_Joints, 1`.
    class_ids : numpy.ndarray or mxnet.nd.NDArray
        Class IDs.
    bboxes : numpy.ndarray or mxnet.nd.NDArray
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
    scores : numpy.ndarray or mxnet.nd.NDArray, optional
        Confidence scores of the provided `bboxes` with shape `N`.
    box_thresh : float, optional, default 0.5
        Display threshold if `scores` is provided. Scores with less than `box_thresh`
        will be ignored in display.
    keypoint_thresh : float, optional, default 0.2
        Keypoints with confidence less than `keypoint_thresh` will be ignored in display.
    scale : float
        The scale of output image, which may affect the positions of boxes

    Returns
    -------
    numpy.ndarray
        The image with estimated pose.

    """
    import matplotlib.pyplot as plt
    cv2 = try_import_cv2()

    if isinstance(img, mx.nd.NDArray):
        img = img.asnumpy()
    if isinstance(coords, mx.nd.NDArray):
        coords = coords.asnumpy()
    if isinstance(class_ids, mx.nd.NDArray):
        class_ids = class_ids.asnumpy()
    if isinstance(bboxes, mx.nd.NDArray):
        bboxes = bboxes.asnumpy()
    if isinstance(scores, mx.nd.NDArray):
        scores = scores.asnumpy()
    if isinstance(confidence, mx.nd.NDArray):
        confidence = confidence.asnumpy()

    joint_visible = confidence[:, :, 0] > keypoint_thresh
    joint_pairs = [[0, 1], [1, 3], [0, 2], [2, 4],
                   [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                   [5, 11], [6, 12], [11, 12],
                   [11, 13], [12, 14], [13, 15], [14, 16]]

    person_ind = class_ids[0] == 0
    #img = cv_plot_bbox(img, bboxes[0][person_ind[:, 0]], scores[0][person_ind[:, 0]],
    #                   thresh=box_thresh, class_names='person', scale=scale, **kwargs)

    colormap_index = np.linspace(0, 1, len(joint_pairs))
    coords *= scale
    for i in range(coords.shape[0]):
        pts = coords[i]
        for cm_ind, jp in zip(colormap_index, joint_pairs):
            if joint_visible[i, jp[0]] and joint_visible[i, jp[1]]:
                cm_color = tuple([int(x * 255) for x in plt.cm.cool(cm_ind)[:3]])
                pt1 = (int(pts[jp, 0][0]), int(pts[jp, 1][0]))
                pt2 = (int(pts[jp, 0][1]), int(pts[jp, 1][1]))
                cv2.line(img, pt1, pt2, cm_color, 2)
    return img


def cv_plot_upper_keypoints(img, coords, confidence, class_ids, bboxes, scores,
                            box_thresh=0.5, keypoint_thresh=0.2, scale=1.0, **kwargs):
    """Visualize keypoints with OpenCV.

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    coords : numpy.ndarray or mxnet.nd.NDArray
        Array with shape `Batch, N_Joints, 2`.
    confidence : numpy.ndarray or mxnet.nd.NDArray
        Array with shape `Batch, N_Joints, 1`.
    class_ids : numpy.ndarray or mxnet.nd.NDArray
        Class IDs.
    bboxes : numpy.ndarray or mxnet.nd.NDArray
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
    scores : numpy.ndarray or mxnet.nd.NDArray, optional
        Confidence scores of the provided `bboxes` with shape `N`.
    box_thresh : float, optional, default 0.5
        Display threshold if `scores` is provided. Scores with less than `box_thresh`
        will be ignored in display.
    keypoint_thresh : float, optional, default 0.2
        Keypoints with confidence less than `keypoint_thresh` will be ignored in display.
    scale : float
        The scale of output image, which may affect the positions of boxes

    Returns
    -------
    numpy.ndarray
        The image with estimated pose.

    """

    if isinstance(img, mx.nd.NDArray):
        img = img.asnumpy()
    if isinstance(coords, mx.nd.NDArray):
        coords = coords.asnumpy()
    if isinstance(class_ids, mx.nd.NDArray):
        class_ids = class_ids.asnumpy()
    if isinstance(bboxes, mx.nd.NDArray):
        bboxes = bboxes.asnumpy()
    if isinstance(scores, mx.nd.NDArray):
        scores = scores.asnumpy()
    if isinstance(confidence, mx.nd.NDArray):
        confidence = confidence.asnumpy()

    joint_visible = confidence[:, :, 0] > keypoint_thresh
    ### 0:"nose",
    ### 1:"left_shoulder", 2:"right_shoulder", 3:"left_elbow", 4:"right_elbow",
    ### 5:"left_wrist", 6:"right_wrist", 7:"left_hip", 8:"right_hip",
    joint_pairs = [[0, 0], [0, 0],
                   [1, 2], [1, 3], [3, 5], [2, 4], [4, 6],
                   [1, 7], [2, 8], [7, 8]
                   ]

    person_ind = class_ids[0] == 0
    #img = cv_plot_bbox(img, bboxes[0][person_ind[:, 0]], scores[0][person_ind[:, 0]],
    #                   thresh=box_thresh, class_names='person', scale=scale, **kwargs)

    colormap_index = np.linspace(0, 1, len(joint_pairs))
    coords *= scale
    for i in range(coords.shape[0]):
        pts = coords[i]
        for cm_ind, jp in zip(colormap_index, joint_pairs):
            if joint_visible[i, jp[0]] and joint_visible[i, jp[1]]:
                cm_color = tuple([int(x * 255) for x in plt.cm.cool(cm_ind)[:3]])
                pt1 = (int(pts[jp, 0][0]), int(pts[jp, 1][0]))
                pt2 = (int(pts[jp, 0][1]), int(pts[jp, 1][1]))
                cv2.line(img, pt1, pt2, cm_color, 3)
    return img


def get_wanted_joints(pred_coords, confidence, wanted_joints=None):
    if wanted_joints is None:
        return pred_coords, confidence

    global KEYPOINTS

    # Create new data
    new_coords = np.zeros((pred_coords.shape[0], len(wanted_joints), 2))
    new_confidence = np.zeros((confidence.shape[0], len(wanted_joints), 1))

    # Select keypoints which are not wanted
    keypoint_ids = []
    for keypoint_id in range(len(KEYPOINTS)):
        keypoint = KEYPOINTS[keypoint_id]
        if keypoint not in wanted_joints:
            keypoint_ids.append(keypoint_id)

    # Delete the selected keypoints
    for cid, (coord, conf) in enumerate(zip(pred_coords, confidence)):
        coord = coord.asnumpy()
        conf = conf.asnumpy()
        new_coords[cid] = np.delete(coord, keypoint_ids, axis=0)
        new_confidence[cid] = np.delete(conf, keypoint_ids, axis=0)
    new_coords = mx.nd.array(new_coords)
    new_confidence = mx.nd.array(new_confidence)

    return new_coords, new_confidence

#utils for normalization
def get_pose_center(left_hip, right_hip):
    return (left_hip+right_hip)*0.5
def get_pose_size(landmarks):
    nose, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip = landmarks
    hips = (left_hip + right_hip) * 0.5
    shoulders = (left_shoulder + right_shoulder) * 0.5
    # Torso size as the minimum body size.
    torso_size = np.linalg.norm(shoulders - hips)
    #print(shoulders, hips)
    #print(torso_size)
    # Max dist to pose center.
    pose_center = get_pose_center(left_hip, right_hip)
    max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))
    return max(torso_size * 2.5, max_dist)
def normalize_pose_landmarks(landmarks):
    """Normalizes landmarks translation and scale."""
    landmarks = np.copy(landmarks)
    nose, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip = landmarks
    # Normalize translation.
    pose_center = get_pose_center(left_hip, right_hip)
    landmarks -= pose_center
    # Normalize scale.
    pose_size = get_pose_size(landmarks)
    landmarks /= pose_size
    # Multiplication by 100 is not required, but makes it eaasier to debug.
    landmarks *= 100
    return landmarks
def calc_feature_pose(landmarks):
    if len(landmarks)>0:
        nose, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip = landmarks
        features = [left_wrist[0], left_wrist[1], right_wrist[0], right_wrist[1]]
        return features
    return None

# utils for filter
from scipy.signal import butter, lfilter
import numpy as np

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff=3.667, fs=30, order=6):
    first = data[0]
    data = np.array(data)
    data -= first
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    y += first
    return y
