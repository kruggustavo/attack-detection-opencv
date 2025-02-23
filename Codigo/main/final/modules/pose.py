import cv2
import numpy as np

from final.modules.keypoints import BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS


class Pose(object):
    num_kpts = 18
    kpt_names = ['nose', 'neck',
                 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                 'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                 'r_eye', 'l_eye',
                 'r_ear', 'l_ear']

    POINTS_LABELS = ["c", "n", "hi", "ci", "mi", "hd", "cd", "md", "hpi", "ri", "pi", "hpd", "rd", "pd", "p"]

    sigmas = np.array([.26, .79, .79, .72, .62, .79, .72, .62, 1.07, .87, .89, 1.07, .87, .89, .25, .25, .35, .35],
                      dtype=np.float32) / 10.0
    vars = (sigmas * 2) ** 2
    last_id = -1
    color = [0, 224, 255]

    def __init__(self, keypoints, confidence):
        super().__init__()
        self.keypoints = keypoints
        self.confidence = confidence
        found_keypoints = np.zeros((np.count_nonzero(keypoints[:, 0] != -1), 2), dtype=np.int32)
        found_kpt_id = 0
        for kpt_id in range(keypoints.shape[0]):
            if keypoints[kpt_id, 0] == -1:
                continue
            found_keypoints[found_kpt_id] = keypoints[kpt_id]
            found_kpt_id += 1
        self.bbox = cv2.boundingRect(found_keypoints)
        self.id = None

    def update_id(self, id=None):
        self.id = id
        if self.id is None:
            self.id = Pose.last_id + 1
            Pose.last_id += 1

    def getOrderedKeypoints(self):
        assert self.keypoints.shape == (Pose.num_kpts, 2)
        finalkeypoints = {}
        for part_id in range(len(BODY_PARTS_PAF_IDS) - 6):
            #
            a = BODY_PARTS_KPT_IDS[part_id][0]
            pa = self.keypoints[a, 0]
            if pa != -1:
                x_a, y_a = self.keypoints[a]
                finalkeypoints[a] = (x_a, y_a)
                #cv2.circle(img, (int(x_a), int(y_a)), 3, Pose.color, -1)
                #cv2.putText(img, str(a), (int(x_a) + 10, int(y_a)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1,  cv2.LINE_AA)

            #
            b = BODY_PARTS_KPT_IDS[part_id][1]
            pb = self.keypoints[b, 0]
            if pb != -1:
                x_b, y_b = self.keypoints[b]
                finalkeypoints[b] = (x_b, y_b)
                #cv2.circle(img, (int(x_b), int(y_b)), 3, Pose.color, -1)
                #cv2.putText(img, str(b), (int(x_b) + 10, int(y_b)), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (180, 180, 180), 1, cv2.LINE_AA)

            #if pa != -1 and pb != -1:
                #cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), Pose.color, 2)

        return finalkeypoints

    def draw(self, img):
        assert self.keypoints.shape == (Pose.num_kpts, 2)

        for part_id in range(len(BODY_PARTS_PAF_IDS) - 6):
            #
            a = BODY_PARTS_KPT_IDS[part_id][0]
            pa = self.keypoints[a, 0]
            if pa != -1:
                x_a, y_a = self.keypoints[a]
                cv2.circle(img, (int(x_a), int(y_a)), 3, Pose.color, -1)
                cv2.putText(img, self.POINTS_LABELS[a], (int(x_a) + 10, int(y_a)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1,  cv2.LINE_AA)

            #
            b = BODY_PARTS_KPT_IDS[part_id][1]
            pb = self.keypoints[b, 0]
            if pb != -1:
                x_b, y_b = self.keypoints[b]
                cv2.circle(img, (int(x_b), int(y_b)), 3, Pose.color, -1)
                cv2.putText(img, self.POINTS_LABELS[b], (int(x_b) + 10, int(y_b)), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (180, 180, 180), 1, cv2.LINE_AA)

            if pa != -1 and pb != -1:
                cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), Pose.color, 2)


def get_similarity(a, b, threshold=0.5):
    num_similar_kpt = 0
    for kpt_id in range(Pose.num_kpts):
        if a.keypoints[kpt_id, 0] != -1 and b.keypoints[kpt_id, 0] != -1:
            distance = np.sum((a.keypoints[kpt_id] - b.keypoints[kpt_id]) ** 2)
            area = max(a.bbox[2] * a.bbox[3], b.bbox[2] * b.bbox[3])
            similarity = np.exp(-distance / (2 * (area + np.spacing(1)) * Pose.vars[kpt_id]))
            if similarity > threshold:
                num_similar_kpt += 1
    return num_similar_kpt


def propagate_ids(previous_poses, current_poses, threshold=3):
    """Propagate poses ids from previous frame results. Id is propagated,
    if there are at least `threshold` similar keypoints between pose from previous frame and current.

    :param previous_poses: poses from previous frame with ids
    :param current_poses: poses from current frame to assign ids
    :param threshold: minimal number of similar keypoints between poses
    :return: None
    """
    current_poses = sorted(current_poses, key=lambda pose: pose.confidence, reverse=True)  # match confident poses first
    mask = np.ones(len(previous_poses), dtype=np.int32)
    for current_pose_id in range(len(current_poses)):
        best_matched_id = None
        best_matched_pose_id = None
        best_matched_iou = 0
        for previous_pose_id in range(len(previous_poses)):
            if not mask[previous_pose_id]:
                continue
            iou = get_similarity(current_poses[current_pose_id], previous_poses[previous_pose_id])
            if iou > best_matched_iou:
                best_matched_iou = iou
                best_matched_pose_id = previous_poses[previous_pose_id].id
                best_matched_id = previous_pose_id
        if best_matched_iou >= threshold:
            mask[best_matched_id] = 0
        else:  # pose not similar to any previous
            best_matched_pose_id = None
        current_poses[current_pose_id].update_id(best_matched_pose_id)
