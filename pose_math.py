import math

import cv2

# RTMPose (via rtmlib's Body class) reports the standard COCO-17 keypoint
# layout: 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
# 5 left_shoulder, 6 right_shoulder, 7-16 elbows/wrists/hips/knees/ankles.
COCO17_NOSE = 0
COCO17_LEFT_EYE = 1
COCO17_RIGHT_EYE = 2
COCO17_LEFT_EAR = 3
COCO17_RIGHT_EAR = 4
COCO17_LEFT_SHOULDER = 5
COCO17_RIGHT_SHOULDER = 6

POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# Useful landmarks for posture
POSTURE_LANDMARKS = {
    "nose": COCO17_NOSE,
    "left_ear": COCO17_LEFT_EAR,
    "right_ear": COCO17_RIGHT_EAR,
    "left_shoulder": COCO17_LEFT_SHOULDER,
    "right_shoulder": COCO17_RIGHT_SHOULDER,
}

MIN_PERSON_PRESENCE_VISIBILITY = 0.6


def extract_landmarks(keypoints, scores):
    """
    Returns selected landmarks in pixel coordinates. RTMPose already reports
    keypoints in pixel space (unlike MediaPipe's normalized 0-1 landmarks),
    so no frame-dimension scaling is needed here.
    px, py are image pixel coordinates.
    visibility is RTMPose's per-keypoint confidence score.
    """
    if len(keypoints) == 0:
        return None

    person_kpts = keypoints[0]
    person_scores = scores[0]
    extracted = {}

    for name, idx in POSTURE_LANDMARKS.items():
        px, py = person_kpts[idx]
        extracted[name] = {
            "px": int(px),
            "py": int(py),
            "visibility": float(person_scores[idx]),
        }

    extracted["neck"] = _estimate_neck(person_kpts, person_scores)

    return extracted


def _estimate_neck(person_kpts, person_scores):
    """
    Neck isn't a native pose-model landmark. Estimate it as the
    visibility-weighted average of both shoulders and both eyes, so an
    occluded/low-confidence shoulder doesn't drag the estimate off to one side.
    """
    NECK_SOURCE_INDICES = (COCO17_LEFT_SHOULDER, COCO17_RIGHT_SHOULDER, COCO17_LEFT_EYE, COCO17_RIGHT_EYE)

    source_points = [person_kpts[i] for i in NECK_SOURCE_INDICES]
    source_scores = [person_scores[i] for i in NECK_SOURCE_INDICES]
    total_weight = sum(source_scores)

    if total_weight <= 0:
        # Fall back to an unweighted average if visibility scores are unusable.
        weights = [1.0] * len(source_points)
        total_weight = float(len(source_points))
    else:
        weights = source_scores

    neck_px = sum(pt[0] * w for pt, w in zip(source_points, weights)) / total_weight
    neck_py = sum(pt[1] * w for pt, w in zip(source_points, weights)) / total_weight
    neck_visibility = sum(source_scores) / len(source_scores)

    return {
        "px": int(neck_px),
        "py": int(neck_py),
        "visibility": neck_visibility,
    }


def is_person_present(selected):
    """
    The person detector occasionally produces a low-confidence detection on a
    non-person object (a chair, a coat on a hook). Requiring decent visibility
    on the core landmarks filters most of those out, since a real person's
    shoulders/nose are confidently detected while a misidentified object's
    keypoints tend to be low-confidence guesses.
    """
    if selected is None:
        return False

    core = (selected["nose"], selected["left_shoulder"], selected["right_shoulder"])
    avg_visibility = sum(lm["visibility"] for lm in core) / len(core)

    return avg_visibility >= MIN_PERSON_PRESENCE_VISIBILITY


def neck_shoulder_angle(selected):
    """
    Interior angle at the neck vertex formed by the segments
    left_shoulder-neck and neck-right_shoulder, in degrees.
    Computed in pixel coordinates so the frame's aspect ratio doesn't skew it.
    """
    neck = selected["neck"]
    left_shoulder = selected["left_shoulder"]
    right_shoulder = selected["right_shoulder"]

    v1 = (left_shoulder["px"] - neck["px"], left_shoulder["py"] - neck["py"])
    v2 = (right_shoulder["px"] - neck["px"], right_shoulder["py"] - neck["py"])

    v1_len = math.hypot(*v1)
    v2_len = math.hypot(*v2)

    if v1_len == 0 or v2_len == 0:
        return None

    cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (v1_len * v2_len)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # guard against float rounding

    return math.degrees(math.acos(cos_angle))


def angle_from_horizontal(point_a, point_b):
    """
    Angle the segment point_a->point_b makes with the horizontal axis, in degrees.
    0 means level, 90 means perfectly vertical. Computed in pixel coordinates.
    """
    dx = point_b["px"] - point_a["px"]
    dy = point_b["py"] - point_a["py"]

    if dx == 0 and dy == 0:
        return None

    return math.degrees(math.atan2(abs(dy), abs(dx)))


def draw_skeleton(frame, keypoints, scores, kpt_thr=0.3):
    if len(keypoints) == 0:
        return

    person_kpts = keypoints[0]
    person_scores = scores[0]

    for start, end in POSE_CONNECTIONS:
        if person_scores[start] < kpt_thr or person_scores[end] < kpt_thr:
            continue
        pt1 = (int(person_kpts[start][0]), int(person_kpts[start][1]))
        pt2 = (int(person_kpts[end][0]), int(person_kpts[end][1]))
        cv2.line(frame, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)
