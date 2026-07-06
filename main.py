import math
import os
import random
import threading
import time
import tkinter as tk

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")

NECK_VERTEX_ALERT_THRESHOLD_DEG = 123
NECK_VERTEX_ALERT_COOLDOWN_SEC = 15
MIN_NOSE_NECK_DISTANCE_PX = 35

BAD_POSTURE_MESSAGES = [
    "I would sit right if I were you!",
    "Do you wanna look like a goblin? cool! keep sitting like that!",
    "Your neck is filing a complaint.",
    "That posture is not giving main character energy.",
    "Sit up before you turn into a question mark.",
]

PoseLandmark = vision.PoseLandmark
POSE_CONNECTIONS = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]

# Useful landmarks for posture
POSTURE_LANDMARKS = {
    "nose": PoseLandmark.NOSE,
    "left_ear": PoseLandmark.LEFT_EAR,
    "right_ear": PoseLandmark.RIGHT_EAR,
    "left_shoulder": PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": PoseLandmark.RIGHT_SHOULDER,
}


def extract_landmarks(result, frame_width, frame_height):
    """
    Returns selected landmarks in both normalized and pixel coordinates.
    x, y are normalized from 0 to 1.
    px, py are actual image pixel coordinates.
    z is relative depth from MediaPipe.
    visibility is confidence-like visibility score.
    """
    if not result.pose_landmarks:
        return None

    landmarks = result.pose_landmarks[0]
    extracted = {}

    for name, landmark_enum in POSTURE_LANDMARKS.items():
        lm = landmarks[landmark_enum.value]

        extracted[name] = {
            "x": lm.x,
            "y": lm.y,
            "z": lm.z,
            "visibility": lm.visibility,
            "px": int(lm.x * frame_width),
            "py": int(lm.y * frame_height),
        }

    extracted["neck"] = _estimate_neck(landmarks, frame_width, frame_height)

    return extracted


def _estimate_neck(landmarks, frame_width, frame_height):
    """
    Neck isn't a native MediaPipe landmark. Estimate it as the visibility-weighted
    average of both shoulders and both eyes, so an occluded/low-confidence shoulder
    doesn't drag the estimate off to one side.
    """
    NECK_SOURCE_LANDMARKS = (
        PoseLandmark.LEFT_SHOULDER,
        PoseLandmark.RIGHT_SHOULDER,
        PoseLandmark.LEFT_EYE,
        PoseLandmark.RIGHT_EYE,
    )

    sources = [landmarks[lm.value] for lm in NECK_SOURCE_LANDMARKS]
    total_weight = sum(lm.visibility for lm in sources)

    if total_weight <= 0:
        # Fall back to an unweighted average if visibility scores are unusable.
        weights = [1.0] * len(sources)
        total_weight = float(len(sources))
    else:
        weights = [lm.visibility for lm in sources]

    neck_x = sum(lm.x * w for lm, w in zip(sources, weights)) / total_weight
    neck_y = sum(lm.y * w for lm, w in zip(sources, weights)) / total_weight
    neck_z = sum(lm.z * w for lm, w in zip(sources, weights)) / total_weight
    neck_visibility = sum(lm.visibility for lm in sources) / len(sources)

    return {
        "x": neck_x,
        "y": neck_y,
        "z": neck_z,
        "visibility": neck_visibility,
        "px": int(neck_x * frame_width),
        "py": int(neck_y * frame_height),
    }


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


def draw_skeleton(frame, result):
    if not result.pose_landmarks:
        return

    landmarks = result.pose_landmarks[0]
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for start, end in POSE_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 255, 255), 1, cv2.LINE_AA)


options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Try 0 first. If you have multiple cameras, try 1 or 2.
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try changing VideoCapture(0) to VideoCapture(1).")

last_print_time = time.time()
last_posture_alert_time = 0.0


def _show_posture_popup(message):
    root = tk.Tk()
    root.title("Posture Alert")
    root.attributes("-topmost", True)
    root.overrideredirect(True)
    root.configure(bg="#1e1e1e")

    width, height = 420, 160
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    root.geometry(f"{width}x{height}+{x}+{y}")

    label = tk.Label(
        root,
        text=message,
        wraplength=380,
        fg="white",
        bg="#1e1e1e",
        font=("Segoe UI", 14, "bold"),
        justify="center",
    )
    label.pack(expand=True, fill="both", padx=20, pady=20)

    root.after(4000, root.destroy)
    root.mainloop()


def alert_bad_posture(vertex_angle):
    message = random.choice(BAD_POSTURE_MESSAGES)
    threading.Thread(target=_show_posture_popup, args=(message,), daemon=True).start()

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame.")
            break

        frame_height, frame_width = frame.shape[:2]

        # OpenCV uses BGR. MediaPipe expects RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Draw full pose skeleton on the frame
        draw_skeleton(frame, result)

        selected = extract_landmarks(result, frame_width, frame_height)

        if selected is not None:
            # Draw only posture-relevant points with labels
            for name, data in selected.items():
                px, py = data["px"], data["py"]

                cv2.circle(frame, (px, py), 6, (0, 255, 0), -1)
                cv2.putText(
                    frame,
                    name,
                    (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            neck = selected["neck"]
            nose = selected["nose"]
            vertex_angle = neck_shoulder_angle(selected)
            left_angle = angle_from_horizontal(selected["left_shoulder"], neck)
            right_angle = angle_from_horizontal(neck, selected["right_shoulder"])
            cva_angle = angle_from_horizontal(neck, nose)
            nose_neck_distance = math.hypot(nose["px"] - neck["px"], nose["py"] - neck["py"])

            if (
                vertex_angle is not None
                and vertex_angle > NECK_VERTEX_ALERT_THRESHOLD_DEG
                and nose_neck_distance >= MIN_NOSE_NECK_DISTANCE_PX
            ):
                now = time.time()
                if now - last_posture_alert_time > NECK_VERTEX_ALERT_COOLDOWN_SEC:
                    alert_bad_posture(vertex_angle)
                    last_posture_alert_time = now

            cv2.line(frame, (selected["left_shoulder"]["px"], selected["left_shoulder"]["py"]), (neck["px"], neck["py"]), (0, 200, 255), 2)
            cv2.line(frame, (neck["px"], neck["py"]), (selected["right_shoulder"]["px"], selected["right_shoulder"]["py"]), (0, 200, 255), 2)
            cv2.line(frame, (neck["px"], neck["py"]), (nose["px"], nose["py"]), (255, 100, 0), 2)

            # Show all angles stacked in the top-left corner
            overlay_lines = []
            if vertex_angle is not None:
                overlay_lines.append(f"neck vertex angle: {vertex_angle:.1f} deg")
            if left_angle is not None:
                overlay_lines.append(f"left shoulder-neck angle: {left_angle:.1f} deg")
            if right_angle is not None:
                overlay_lines.append(f"neck-right shoulder angle: {right_angle:.1f} deg")
            if cva_angle is not None:
                overlay_lines.append(f"craniovertebral angle: {cva_angle:.1f} deg")
            overlay_lines.append(f"nose-neck distance: {nose_neck_distance:.1f} px")

            for i, line in enumerate(overlay_lines):
                cv2.putText(
                    frame,
                    line,
                    (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

            # Print extracted values once per second
            now = time.time()
            if now - last_print_time > 1.0:
                print("\nSelected posture landmarks:")
                for name, data in selected.items():
                    print(
                        f"{name:15s} "
                        f"x={data['x']:.3f}, y={data['y']:.3f}, z={data['z']:.3f}, "
                        f"px={data['px']:4d}, py={data['py']:4d}, "
                        f"visibility={data['visibility']:.2f}"
                    )
                for line in overlay_lines:
                    print(line)
                last_print_time = now

        cv2.imshow("Live Pose Landmark Extraction", frame)

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
