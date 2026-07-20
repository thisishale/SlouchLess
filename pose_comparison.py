"""
Standalone tool to visually compare MediaPipe's pose landmark detection
against three other pose-estimation models on the same live webcam feed.
Not part of the SlouchLess app - a one-off dev/testing script for judging
which model tracks posture-relevant landmarks (neck/shoulders/head) best.

Opens 4 windows side by side:
  - MediaPipe Pose      (mediapipe, Google)
  - YOLO-Pose           (ultralytics)
  - MoveNet             (tensorflow-hub, Google)
  - RTMPose             (rtmlib, OpenMMLab)

Press 'q' in any window to quit.
"""

import os
import time

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from rtmlib import Body, draw_skeleton as rtm_draw_skeleton
from ultralytics import YOLO

CAMERA_INDEX = 0
WINDOW_NAMES = ["MediaPipe", "YOLO-Pose", "MoveNet", "RTMPose"]

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_lite.task")

MOVENET_INPUT_SIZE = 192
MOVENET_KEYPOINT_THRESHOLD = 0.3
# Standard COCO-17 keypoint skeleton edges, used by MoveNet's output order.
MOVENET_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def build_mediapipe_landmarker():
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def run_mediapipe(landmarker, frame, timestamp_ms):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    annotated = frame.copy()
    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]
        h, w = frame.shape[:2]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        connections = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]
        for start, end in connections:
            cv2.line(annotated, points[start], points[end], (0, 255, 0), 2, cv2.LINE_AA)
        for x, y in points:
            cv2.circle(annotated, (x, y), 3, (0, 0, 255), -1)
    return annotated


def run_yolo_pose(model, frame):
    results = model(frame, verbose=False)
    return results[0].plot()


def run_movenet(movenet, frame):
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = tf.image.resize_with_pad(np.expand_dims(rgb_frame, axis=0), MOVENET_INPUT_SIZE, MOVENET_INPUT_SIZE)
    img = tf.cast(img, dtype=tf.int32)

    outputs = movenet(img)
    keypoints = outputs["output_0"].numpy()[0, 0]  # (17, 3): normalized y, x, score

    points = [(int(x * w), int(y * h), score) for y, x, score in keypoints]

    annotated = frame.copy()
    for start, end in MOVENET_EDGES:
        x1, y1, s1 = points[start]
        x2, y2, s2 = points[end]
        if s1 > MOVENET_KEYPOINT_THRESHOLD and s2 > MOVENET_KEYPOINT_THRESHOLD:
            cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
    for x, y, score in points:
        if score > MOVENET_KEYPOINT_THRESHOLD:
            cv2.circle(annotated, (x, y), 3, (0, 0, 255), -1)
    return annotated


def run_rtmpose(body, frame):
    keypoints, scores = body(frame)
    return rtm_draw_skeleton(frame.copy(), keypoints, scores, kpt_thr=0.5)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Loading models...")
    mp_landmarker = build_mediapipe_landmarker()
    yolo_model = YOLO("yolov8n-pose.pt")
    movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4").signatures["serving_default"]
    rtm_body = Body(mode="lightweight", backend="onnxruntime", device="cpu")
    print("Models loaded. Press 'q' in any window to quit.")

    for i, name in enumerate(WINDOW_NAMES):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 480, 360)
        cv2.moveWindow(name, (i % 2) * 500, (i // 2) * 420)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame.")
            break

        timestamp_ms = int(time.time() * 1000)

        cv2.imshow("MediaPipe", run_mediapipe(mp_landmarker, frame, timestamp_ms))
        cv2.imshow("YOLO-Pose", run_yolo_pose(yolo_model, frame))
        cv2.imshow("MoveNet", run_movenet(movenet, frame))
        cv2.imshow("RTMPose", run_rtmpose(rtm_body, frame))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
