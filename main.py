import contextlib
import ctypes
import glob
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import threading
import time
import tkinter as tk
import traceback
from tkinter import messagebox

import cv2
import joblib
import pandas as pd
from rtmlib import Body

IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    import winsound
    from pycaw.pycaw import AudioUtilities
    from pygrabber.dshow_graph import FilterGraph


def _show_fatal_error(error_text):
    """
    Shows `error_text` in a Tkinter error dialog. The app is built with
    console=False, so an uncaught exception's traceback would otherwise just
    vanish - there's no terminal attached to print it to when launched by
    double-clicking the executable.
    """
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("SlouchLess - Fatal Error", error_text)
        root.destroy()
    except Exception:
        pass  # if Tk itself is broken, there's nothing left to show it with


def _handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    error_text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print(error_text, file=sys.stderr)
    _show_fatal_error(error_text)


def _handle_uncaught_thread_exception(args):
    _handle_uncaught_exception(args.exc_type, args.exc_value, args.exc_traceback)


sys.excepthook = _handle_uncaught_exception
threading.excepthook = _handle_uncaught_thread_exception


def resource_path(*parts):
    """
    Resolves a path to a bundled, read-only resource (model file, icon image).
    Works both running from source and when frozen into a PyInstaller exe,
    where bundled data is extracted under sys._MEIPASS instead of living next
    to the script.
    """
    base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, *parts)


def set_window_icon(window_title, icon_path):
    """
    OpenCV's highgui windows use a hardcoded icon baked into its own DLL and
    expose no API to change it, so the title-bar/taskbar icon has to be set
    directly via the Win32 API on the specific window instance instead.
    """
    IMAGE_ICON = 1
    LR_LOADFROMFILE = 0x00000010
    WM_SETICON = 0x0080
    ICON_SMALL = 0
    ICON_BIG = 1

    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_title)
        if not hwnd:
            return
        hicon_big = user32.LoadImageW(None, icon_path, IMAGE_ICON, 32, 32, LR_LOADFROMFILE)
        hicon_small = user32.LoadImageW(None, icon_path, IMAGE_ICON, 16, 16, LR_LOADFROMFILE)
        if hicon_big:
            user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon_big)
        if hicon_small:
            user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon_small)
    except (AttributeError, OSError):
        pass


CALIBRATION_ICON_PATH = resource_path("images", "SlouchImageopt.png")
APP_ICON_PATH = resource_path("images", "SlouchLess.ico")
SLOUCH_MODEL_PATH = resource_path("models", "slouch_classifier.joblib")
MODEL_TYPE_LABELS = {
    "mlp": "Neural Net (MLP)",
    "svm_rbf": "SVM (RBF kernel)",
    "random_forest": "Random Forest",
}
# ask_choice() returns the chosen key verbatim, so this needs to be a value
# that can never collide with an entry in MODEL_TYPE_LABELS.
THRESHOLDS_CHOICE_KEY = "__calibrated_thresholds__"
SWITCH_MODEL_BUTTON_LABEL = "Switch Model"
VIDEO_WINDOW_NAME = "SlouchLess"
# default is 123.
NECK_VERTEX_ALERT_THRESHOLD_DEG = 123
NECK_VERTEX_ALERT_COOLDOWN_SEC = 15
SUSTAINED_BAD_POSTURE_SEC = 5
GOOD_POSTURE_GRACE_SEC = 1
CVA_ANGLE_ALERT_THRESHOLD_DEG = 85.0
NOSE_NECK_DISTANCE_ALERT_THRESHOLD_PX = 60.0
USE_NECK_VERTEX_ANGLE = True
NO_PERSON_DISABLE_TIMEOUT_SEC = 30
MIN_PERSON_PRESENCE_VISIBILITY = 0.6

if IS_WINDOWS:
    CALIBRATION_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "SlouchLess")
else:
    CALIBRATION_DIR = os.path.join(os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")), "SlouchLess")
os.makedirs(CALIBRATION_DIR, exist_ok=True)
CALIBRATION_FILE = os.path.join(CALIBRATION_DIR, "calibration.json")
UPRIGHT_STEP_RECORDING_SEC = 10
SLOUCH_STEP_RECORDING_SEC = 10
CALIBRATION_WINDOW_NAME = "Posture Calibration"
CALIBRATION_PREP_COUNTDOWN_SEC = 5
BEEP_FREQUENCY_HZ = 800
BEEP_DURATION_MS = 150
LOW_VOLUME_THRESHOLD = 0.10

BAD_POSTURE_MESSAGES = [
    "I would sit right if I were you!",
    "Do you wanna look like a goblin? cool! keep sitting like that!",
    "Your neck is filing a complaint.",
    "Cringe!",
    "Wanna turn into a question mark.",
]


def _get_linux_volume_state():
    """
    Reads the default sink's mute/volume state via `pactl` (PulseAudio/
    PipeWire's CLI, present on virtually all Linux desktops). Raises on any
    unexpected output/missing binary - the caller treats that as "can't tell,
    don't warn", same as a Windows pycaw failure.
    """
    mute_output = subprocess.run(
        ["pactl", "get-sink-mute", "@DEFAULT_SINK@"], capture_output=True, text=True, timeout=2, check=True
    ).stdout
    is_muted = "yes" in mute_output.lower()

    volume_output = subprocess.run(
        ["pactl", "get-sink-volume", "@DEFAULT_SINK@"], capture_output=True, text=True, timeout=2, check=True
    ).stdout
    match = re.search(r"(\d+)%", volume_output)
    volume_level = int(match.group(1)) / 100

    return is_muted, volume_level


def check_volume_and_warn(calib_window, mute_threshold=LOW_VOLUME_THRESHOLD):
    """
    Warns the user via `calib_window` if system audio is muted or very quiet,
    since calibration relies on beep sounds to guide each step. Fails open
    (no warning) if the audio state can't be read for any reason. Either way
    (volume fine, or user clicks OK after being warned) calibration proceeds
    regardless of the actual volume level - this is just a heads-up.

    Returns False only if the window was closed while showing the warning,
    so the caller can abort calibration; True otherwise.
    """
    try:
        if IS_WINDOWS:
            volume = AudioUtilities.GetSpeakers().EndpointVolume
            is_muted = bool(volume.GetMute())
            volume_level = volume.GetMasterVolumeLevelScalar()
        else:
            is_muted, volume_level = _get_linux_volume_state()
    except Exception:
        return True

    if not (is_muted or volume_level < mute_threshold):
        return True

    reason = "muted" if is_muted else f"very low ({volume_level * 100:.0f}%)"
    message = (
        f"Your system volume appears to be {reason}.\n\n"
        "Calibration uses beep sounds to guide you through each step - "
        "please turn up your volume before continuing."
    )

    return calib_window.wait_for_start(message, button_text="OK")


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


def compute_ml_feature_row(selected, vertex_angle, cva_angle, shoulder_tilt_angle, ear_tilt_angle, is_camera_above):
    """
    Assembles the same relative, camera-distance-invariant feature set
    train_slouch_classifier.py trains its model on, reusing the angles
    already computed per-frame for the on-screen overlay rather than
    recomputing them. Returns None if any of those angles couldn't be
    computed (e.g. overlapping landmarks), so the caller can skip ML
    inference for that frame instead of feeding the model garbage.
    """
    if None in (vertex_angle, cva_angle, shoulder_tilt_angle, ear_tilt_angle):
        return None

    left_shoulder = selected["left_shoulder"]
    right_shoulder = selected["right_shoulder"]
    shoulder_width = max(
        math.hypot(right_shoulder["px"] - left_shoulder["px"], right_shoulder["py"] - left_shoulder["py"]), 1.0
    )
    nose = selected["nose"]
    neck = selected["neck"]
    nose_neck_dist_norm = math.hypot(nose["px"] - neck["px"], nose["py"] - neck["py"]) / shoulder_width
    left_ear = selected["left_ear"]
    right_ear = selected["right_ear"]
    left_ear_shoulder_dist = math.hypot(left_ear["px"] - left_shoulder["px"], left_ear["py"] - left_shoulder["py"])
    right_ear_shoulder_dist = math.hypot(right_ear["px"] - right_shoulder["px"], right_ear["py"] - right_shoulder["py"])
    head_tuck_dist_norm = (left_ear_shoulder_dist + right_ear_shoulder_dist) / 2 / shoulder_width
    # Plain sum()/len() rather than statistics.mean(): the neck landmark's
    # visibility is a numpy.float32 (from _estimate_neck's average of raw
    # model scores) while the others are plain floats, and statistics.mean's
    # strict type coercion rejects that mix.
    visibilities = [lm["visibility"] for lm in selected.values()]
    avg_visibility = sum(visibilities) / len(visibilities)

    return {
        "vertex_angle_deg": vertex_angle,
        "cva_angle_deg": cva_angle,
        "nose_neck_dist_norm": nose_neck_dist_norm,
        "shoulder_tilt_deg": shoulder_tilt_angle,
        "ear_tilt_deg": ear_tilt_angle,
        "head_tuck_dist_norm": head_tuck_dist_norm,
        "avg_visibility": avg_visibility,
        "camera_position_above": 1.0 if is_camera_above else 0.0,
        "camera_position_below": 0.0 if is_camera_above else 1.0,
        "camera_position_unknown": 0.0,
    }


def predict_slouch(slouch_model, feature_row):
    """
    Returns True/False for whether `feature_row` is predicted as slouching.
    Passed as a one-row DataFrame using the model's own trained-on column
    names/order (from the loaded bundle, not hardcoded here), since the
    model was fit on a DataFrame and predicting from a plain array instead
    triggers a sklearn UserWarning about missing feature names.
    """
    feature_columns = slouch_model["feature_columns"]
    row_df = pd.DataFrame([feature_row], columns=feature_columns)
    return bool(slouch_model["model"].predict(row_df)[0])


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


def load_calibration_settings(path, default_threshold, default_cva_threshold, default_distance_threshold):
    """
    Loads previously saved calibration values from disk. Each value falls back
    independently to its default if the file is missing, unparsable, or only
    has some of the keys.
    """
    if not os.path.exists(path):
        return default_threshold, default_cva_threshold, default_distance_threshold

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        print(f"Could not read calibration file at {path}, using default values.")
        return default_threshold, default_cva_threshold, default_distance_threshold

    try:
        threshold = float(data["neck_vertex_alert_threshold_deg"])
    except (KeyError, ValueError):
        threshold = default_threshold

    try:
        cva_threshold = float(data["cva_angle_alert_threshold_deg"])
    except (KeyError, ValueError):
        cva_threshold = default_cva_threshold

    try:
        distance_threshold = float(data["nose_neck_distance_alert_threshold_px"])
    except (KeyError, ValueError):
        distance_threshold = default_distance_threshold

    return threshold, cva_threshold, distance_threshold


def save_calibration_settings(path, threshold, cva_threshold, distance_threshold, debug_stats=None):
    """
    `debug_stats`, if given, is written under a separate key purely for the
    user's own reference - it's never read back by load_calibration_settings.
    """
    data = {
        "neck_vertex_alert_threshold_deg": threshold,
        "cva_angle_alert_threshold_deg": cva_threshold,
        "nose_neck_distance_alert_threshold_px": distance_threshold,
    }
    if debug_stats is not None:
        data["calibration_debug_stats"] = debug_stats

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(
        f"Saved calibrated threshold ({threshold:.1f} deg), CVA angle threshold ({cva_threshold:.1f} deg), "
        f"and nose-neck distance threshold ({distance_threshold:.1f} px) to {path}"
    )


NECK_VERTEX_ALERT_THRESHOLD_DEG, CVA_ANGLE_ALERT_THRESHOLD_DEG, NOSE_NECK_DISTANCE_ALERT_THRESHOLD_PX = load_calibration_settings(
    CALIBRATION_FILE, NECK_VERTEX_ALERT_THRESHOLD_DEG, CVA_ANGLE_ALERT_THRESHOLD_DEG, NOSE_NECK_DISTANCE_ALERT_THRESHOLD_PX
)


def load_slouch_model(path):
    """
    Loads the classifier trained by train_slouch_classifier.py, if one has
    been trained yet. Returns None (the caller then falls back to the
    calibrated angle/distance thresholds) if no model file exists or it
    fails to load for any reason - training a model is optional, so its
    absence isn't an error.
    """
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Could not load slouch model at {path} ({e}); using calibrated thresholds instead.")
        return None


def resolve_slouch_model_path(is_camera_above):
    """
    Fallback for when discover_available_models() finds no per-type .joblib
    files at all (e.g. only the older combined-model naming scheme was ever
    trained). Prefers a model trained specifically for this camera position
    (via `train_slouch_classifier.py --camera-position above` or `below`),
    since that's a real feature the model was trained on. Falls back to the
    combined "all positions" model if no position-specific one has been
    trained yet.
    """
    position = "above" if is_camera_above else "below"
    position_specific_path = resource_path("models", f"slouch_classifier_{position}.joblib")
    if os.path.exists(position_specific_path):
        return position_specific_path
    return SLOUCH_MODEL_PATH


def discover_available_models(position):
    """
    Looks up each model type in MODEL_TYPE_LABELS (kept in sync with
    train_slouch_classifier.py's --model choices) under the filename
    convention model_file_for() there uses: models/slouch_classifier_
    {position}_{model_name}.joblib. Only types whose .joblib actually exists
    on disk are returned, so the picker never offers a model that isn't
    there to load.

    Returns {model_name: model_file_path}, in MODEL_TYPE_LABELS order.
    """
    available = {}
    for model_name in MODEL_TYPE_LABELS:
        model_file = resource_path("models", f"slouch_classifier_{position}_{model_name}.joblib")
        if os.path.exists(model_file):
            available[model_name] = model_file
    return available


def prompt_model_switch(calib_window, available_models):
    """
    Reopens the shared calibration root (hidden, not destroyed, since the
    startup dialogs finished) to let the model driving live inference be
    changed mid-session from the "Switch Model" button on the video window.
    Reuses that one root rather than creating a second Tk() - Tcl only
    tolerates one live root/thread at a time, and a second one here would
    race it exactly the way alert_bad_posture's old threaded popup did (see
    its docstring). With exit_on_close now False, dismissing this dialog via
    its own window-manager close button just cancels it (hides, doesn't
    destroy) - same outcome as clicking no button at all. Returns
    (model_name, slouch_model) for the user's pick, or None if they closed
    the dialog or picked the model already in use.
    """
    if calib_window.closed:
        # Only reachable if the shared root itself broke (e.g. a TclError
        # during pump()), not from the user just closing this dialog.
        print("Calibration window is no longer available; can't switch models.")
        return None

    choices = [(name, MODEL_TYPE_LABELS.get(name, name)) for name in MODEL_TYPE_LABELS if name in available_models]
    choices.append((THRESHOLDS_CHOICE_KEY, "Calibrated thresholds"))

    calib_window.reopen()
    try:
        chosen = calib_window.ask_choice("Select which trained model to use:", choices)
    finally:
        if not calib_window.closed:
            calib_window.hide()

    if chosen is None:
        return None
    if chosen == THRESHOLDS_CHOICE_KEY:
        return (None, None)
    return (chosen, load_slouch_model(available_models[chosen]))


def _v4l2_supports_capture(device_path):
    """
    Queries whether `device_path` is an actual video capture node via the
    VIDIOC_QUERYCAP ioctl. Some UVC webcams expose a second, metadata-only
    /dev/videoN node right next to their real capture node, sharing the same
    sysfs name - this filters those out of the camera picker so the same
    camera doesn't show up twice. Returns True on any failure to query, so a
    device that can't be inspected still shows up rather than silently
    disappearing.
    """
    import fcntl
    import struct

    V4L2_CAP_VIDEO_CAPTURE = 0x00000001
    V4L2_CAP_DEVICE_CAPS = 0x80000000
    VIDIOC_QUERYCAP = 0x80685600

    try:
        with open(device_path, "rb") as f:
            buf = bytearray(104)  # sizeof(struct v4l2_capability)
            fcntl.ioctl(f.fileno(), VIDIOC_QUERYCAP, buf, True)
            capabilities, device_caps = struct.unpack_from("=II", buf, 84)
            if capabilities & V4L2_CAP_DEVICE_CAPS:
                capabilities = device_caps
            return bool(capabilities & V4L2_CAP_VIDEO_CAPTURE)
    except OSError:
        return True


def list_camera_names():
    """
    Returns detected cameras as (index, name) pairs, where `index` is the
    value to pass to cv2.VideoCapture. On Windows this matches DirectShow's
    enumeration order. On Linux it's read directly from each /dev/videoN
    node's sysfs entry instead of assumed to be contiguous, filtering out
    non-capture nodes via _v4l2_supports_capture - a physical webcam commonly
    exposes a second, metadata-only node right next to its real one sharing
    the same name, and position-in-list can't be trusted to equal the device
    index the way it can on Windows. Returns [] if enumeration fails entirely.
    """
    if IS_WINDOWS:
        try:
            return list(enumerate(FilterGraph().get_input_devices()))
        except Exception:
            return []

    cameras = []
    for device_path in sorted(glob.glob("/dev/video*")):
        try:
            index = int(device_path.removeprefix("/dev/video"))
        except ValueError:
            continue
        if not _v4l2_supports_capture(device_path):
            continue
        try:
            with open(f"/sys/class/video4linux/video{index}/name", "r") as f:
                name = f.read().strip()
        except OSError:
            name = f"Camera {index}"
        cameras.append((index, name))
    return cameras


def prompt_for_camera(cameras):
    """
    Shows a window listing detected camera names as buttons and blocks until
    one is clicked. Returns the chosen device index, or None if the window
    was closed without a selection.
    """
    root = tk.Tk()
    root.title("SlouchLess - Select Camera")
    root.attributes("-topmost", True)

    try:
        icon_image = tk.PhotoImage(file=CALIBRATION_ICON_PATH)
        root.iconphoto(True, icon_image)
    except tk.TclError:
        pass

    chosen = {"index": None}

    def choose(index):
        chosen["index"] = index
        root.destroy()

    tk.Label(root, text="Select which camera to use:", font=("Segoe UI", 12)).pack(padx=20, pady=(20, 10))
    for index, name in cameras:
        tk.Button(root, text=name, width=30, command=lambda i=index: choose(i)).pack(padx=20, pady=4)
    tk.Frame(root, height=10).pack()

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.update_idletasks()
    width = root.winfo_reqwidth()
    height = root.winfo_reqheight()
    x = (root.winfo_screenwidth() - width) // 2
    y = (root.winfo_screenheight() - height) // 2
    root.geometry(f"{width}x{height}+{x}+{y}")

    root.mainloop()
    return chosen["index"]


cameras = list_camera_names()
if not cameras:
    messagebox.showerror("SlouchLess", "No camera was detected. The app will now close.")
    sys.exit(1)

camera_index = prompt_for_camera(cameras)
if camera_index is None:
    sys.exit(0)

cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_V4L2)

if not cap.isOpened():
    raise RuntimeError("Could not open the selected webcam.")

last_print_time = time.time()
last_posture_alert_time = 0.0
last_person_seen_time = time.time()
monitoring_enabled = True
bad_posture_since = None
good_posture_since = None


def _show_posture_popup(calib_window, message):
    """
    Shows a transient "you're slouching" toast as a Toplevel on the shared
    calibration root, rather than spawning its own thread + Tk() root. Tcl
    only tolerates being touched from one thread at a time - a second root
    from a background thread here can race the shared root's own dialogs
    (e.g. the switch-model picker) and crash with
    "Tcl_AsyncDelete: async handler deleted by the wrong thread". Nothing
    here runs its own mainloop(); the popup only actually renders and
    auto-dismisses via the main loop's regular calib_window.pump() calls.
    """
    if calib_window.closed:
        return

    popup = tk.Toplevel(calib_window.root)
    popup.attributes("-topmost", True)
    popup.overrideredirect(True)
    popup.configure(bg="#1e1e1e")

    label = tk.Label(
        popup,
        text=message,
        wraplength=380,
        fg="white",
        bg="#1e1e1e",
        font=("Segoe UI", 14, "bold"),
        justify="center",
    )
    label.pack(expand=True, fill="both", padx=20, pady=20)

    # Size the window to fit the message instead of a fixed geometry, so a
    # longer BAD_POSTURE_MESSAGES entry that wraps to more lines doesn't get
    # clipped by a too-short fixed height.
    popup.update_idletasks()
    width = popup.winfo_reqwidth()
    height = popup.winfo_reqheight()
    x = (popup.winfo_screenwidth() - width) // 2
    y = (popup.winfo_screenheight() - height) // 2
    popup.geometry(f"{width}x{height}+{x}+{y}")

    popup.after(4000, popup.destroy)


def alert_bad_posture(calib_window, vertex_angle):
    message = random.choice(BAD_POSTURE_MESSAGES)
    _show_posture_popup(calib_window, message)


class CalibrationWindow:
    """
    A single persistent Tkinter window that hosts all calibration dialog
    (the calibrate y/n prompt, phase transitions, step instructions/countdown,
    results, and the save-permanently prompt), so none of that text ends up on
    the console or drawn onto the webcam feed.

    Uses root.update() in a polling loop instead of root.mainloop(), so it can
    run interleaved with the OpenCV capture loop on the same thread.
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SlouchLess Calibration")
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        try:
            # Keep a reference on self - Tk doesn't hold one, so the image
            # would otherwise get garbage collected and the icon would vanish.
            self._icon_image = tk.PhotoImage(file=CALIBRATION_ICON_PATH)
            self.root.iconphoto(True, self._icon_image)
        except tk.TclError:
            pass

        self.message_label = tk.Label(self.root, text="", wraplength=380, justify="center", font=("Segoe UI", 12))
        self.message_label.pack(expand=True, fill="both", padx=20, pady=(20, 10))

        self.countdown_label = tk.Label(self.root, text="", font=("Segoe UI", 22, "bold"))
        self.countdown_label.pack(pady=(0, 10))

        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=(0, 15))

        self._response = None
        self.closed = False
        self.closed_by_user = False
        self._destroyed = False
        self._cancelled = False
        # True during the startup dialogs, where closing the window means the
        # user is opting out entirely (main() exits). Set False once startup
        # finishes and this window becomes the shared, session-long root
        # reused for mid-session prompts (switch-model) and alert popups -
        # closing it there should just cancel that one prompt, not tear down
        # the shared root the rest of the session depends on.
        self.exit_on_close = True

    def _on_close(self):
        self._cancelled = True
        if self.exit_on_close:
            self.closed_by_user = True
            self.close()
        else:
            self.hide()

    def pump(self):
        if self.closed:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.closed = True
            self._destroyed = True

    def _autosize(self):
        """
        Resizes/recenters the window to fit its current content instead of a
        fixed geometry, so switching between short instructions and the much
        longer multi-line results text doesn't squeeze the buttons. Not
        called from set_countdown(), since that fires every frame and would
        make the window jitter as digit count changes (e.g. "30s" -> "5s").
        """
        if self.closed:
            return
        try:
            self.root.update_idletasks()
            width = self.root.winfo_reqwidth()
            height = self.root.winfo_reqheight()
            x = (self.root.winfo_screenwidth() - width) // 2
            y = (self.root.winfo_screenheight() - height) // 2
            self.root.geometry(f"{width}x{height}+{x}+{y}")
        except tk.TclError:
            self.closed = True
            self._destroyed = True

    def set_message(self, text):
        self.message_label.config(text=text)
        self._autosize()
        self.pump()

    def set_countdown(self, text):
        self.countdown_label.config(text=text)
        self.pump()

    def _clear_buttons(self):
        for widget in self.button_frame.winfo_children():
            widget.destroy()

    def clear_buttons(self):
        """Removes any buttons left over from a prior ask_yes_no()/wait_for_start() and resizes to fit."""
        self._clear_buttons()
        self._autosize()

    def _wait_for_response(self):
        self._cancelled = False
        while self._response is None and not self.closed and not self._cancelled:
            self.pump()
            time.sleep(0.01)
        return self._response

    def ask_yes_no(self, message, yes_text="Yes", no_text="No"):
        """Shows `message` with Yes/No buttons and blocks until clicked. Returns True/False."""
        self.set_message(message)
        self.countdown_label.config(text="")
        self._clear_buttons()
        self._response = None

        tk.Button(self.button_frame, text=yes_text, width=10, command=lambda: setattr(self, "_response", True)).pack(
            side="left", padx=10
        )
        tk.Button(self.button_frame, text=no_text, width=10, command=lambda: setattr(self, "_response", False)).pack(
            side="left", padx=10
        )
        self._autosize()

        return bool(self._wait_for_response())

    def ask_choice(self, message, choices):
        """
        Shows `message` with one button per (key, label) pair in `choices`
        (in order) and blocks until one is clicked. Returns the chosen key,
        or None if the window was closed without a selection.
        """
        self.set_message(message)
        self.countdown_label.config(text="")
        self._clear_buttons()
        self._response = None

        for key, label in choices:
            tk.Button(
                self.button_frame, text=label, width=30, command=lambda k=key: setattr(self, "_response", k)
            ).pack(padx=10, pady=4)
        self._autosize()

        return self._wait_for_response()

    def wait_for_start(self, message, button_text="Start"):
        """Shows `message` with a single button and blocks until clicked. Returns True/False (False if window closed)."""
        self.set_message(message)
        self.countdown_label.config(text="")
        self._clear_buttons()
        self._response = None

        tk.Button(self.button_frame, text=button_text, width=10, command=lambda: setattr(self, "_response", True)).pack()
        self._autosize()

        return bool(self._wait_for_response())

    def close(self):
        if self._destroyed:
            return
        self._destroyed = True
        self.closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def hide(self):
        """
        Withdraws the window without destroying it, so it can be reopen()'d
        later for a mid-session prompt (e.g. switching models) without
        creating a second Tk() root - Tcl only tolerates one per process, and
        a second one racing this one is what produces the
        Tcl_AsyncDelete "wrong thread" crash.
        """
        if self.closed:
            return
        try:
            self.root.withdraw()
        except tk.TclError:
            self.closed = True
            self._destroyed = True

    def reopen(self):
        """Redisplays a window previously hidden via hide()."""
        if self.closed:
            return
        try:
            self.root.deiconify()
            self.root.attributes("-topmost", True)
        except tk.TclError:
            self.closed = True
            self._destroyed = True


def _play_beep(calib_window):
    if IS_WINDOWS:
        winsound.Beep(BEEP_FREQUENCY_HZ, BEEP_DURATION_MS)
        return

    # winsound.Beep has no non-Windows equivalent. Tk's bell() rings the
    # system bell (XBell on X11) with zero extra dependencies beyond Tk
    # itself, which the app already bundles for its UI - unlike shelling out
    # to an audio library, nothing extra needs to be installed on the end
    # user's machine.
    calib_window.root.bell()


def _read_and_detect(cap, landmarker):
    """
    Reads one frame, runs pose detection, draws the skeleton, and extracts the
    posture landmarks. Returns (frame, selected), or (None, None) if the camera
    read failed.
    """
    ret, frame = cap.read()
    if not ret:
        return None, None

    keypoints, scores = landmarker(frame)

    draw_skeleton(frame, keypoints, scores)
    selected = extract_landmarks(keypoints, scores)

    return frame, selected


def _run_calibration_countdown(cap, landmarker, calib_window, phase_title, instruction, duration_sec):
    """
    Shows `instruction` as an upcoming step with a "get ready" countdown in
    `calib_window`. Records nothing. The webcam feed shows only the skeleton,
    no text. Returns False if aborted with 'q', the window was closed, or the
    camera read failed, True once the countdown elapses.
    """
    start_time = time.time()

    while True:
        remaining = duration_sec - (time.time() - start_time)
        if remaining <= 0:
            return True

        frame, _ = _read_and_detect(cap, landmarker)
        if frame is None:
            return False

        cv2.imshow(CALIBRATION_WINDOW_NAME, frame)

        calib_window.set_message(f"{phase_title}\n\nNext: {instruction}\n\nGet ready...")
        calib_window.set_countdown(f"{remaining:0.0f}s")

        if calib_window.closed:
            return False

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

        if cv2.getWindowProperty(CALIBRATION_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return False


def _run_calibration_recording(cap, landmarker, calib_window, phase_title, instruction, duration_sec, angles, cvas, distances):
    """
    Shows `instruction` as the active step with a countdown in `calib_window`,
    appending the neck vertex angle, craniovertebral angle, and nose-neck
    distance of every confidently detected frame to `angles`/`cvas`/`distances`.
    The webcam feed shows only the skeleton, no text. Returns False if
    aborted with 'q', the window was closed, or the camera read failed, True
    once the countdown elapses.
    """
    start_time = time.time()

    while True:
        remaining = duration_sec - (time.time() - start_time)
        if remaining <= 0:
            return True

        frame, selected = _read_and_detect(cap, landmarker)
        if frame is None:
            return False

        if is_person_present(selected):
            vertex_angle = neck_shoulder_angle(selected)
            if vertex_angle is not None:
                angles.append(vertex_angle)

            nose = selected["nose"]
            neck = selected["neck"]
            cva_angle = angle_from_horizontal(neck, nose)
            if cva_angle is not None:
                cvas.append(cva_angle)

            distances.append(math.hypot(nose["px"] - neck["px"], nose["py"] - neck["py"]))

        cv2.imshow(CALIBRATION_WINDOW_NAME, frame)

        calib_window.set_message(f"{phase_title}\n\n{instruction}\n\nRecording...")
        calib_window.set_countdown(f"{remaining:0.0f}s")

        if calib_window.closed:
            return False

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

        if cv2.getWindowProperty(CALIBRATION_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return False


def run_calibration_phase(cap, landmarker, calib_window, phase_title, steps, step_duration_sec):
    """
    For each instruction in `steps`: beep and show it with a
    CALIBRATION_PREP_COUNTDOWN_SEC "get ready" countdown (not recorded), then
    beep again and record the neck vertex angle, craniovertebral angle, and
    nose-neck distance for step_duration_sec seconds. `step_duration_sec` is
    a required, explicit per-step length rather than a total split across
    len(steps), so changing how many steps a phase has doesn't silently
    change each step's length.

    Returns (angles_by_step, cvas_by_step, distances_by_step), each a list of
    per-frame lists in the same order as `steps`. Returns (None, None, None)
    if any step was aborted with 'q', the window was closed, or no person
    was ever detected. Reuses the already-open `cap`/`landmarker`.
    """
    angles_by_step = [[] for _ in steps]
    cvas_by_step = [[] for _ in steps]
    distances_by_step = [[] for _ in steps]

    for step_index, instruction in enumerate(steps):
        _play_beep(calib_window)
        if not _run_calibration_countdown(cap, landmarker, calib_window, phase_title, instruction, CALIBRATION_PREP_COUNTDOWN_SEC):
            return None, None, None

        _play_beep(calib_window)
        if not _run_calibration_recording(
            cap,
            landmarker,
            calib_window,
            phase_title,
            instruction,
            step_duration_sec,
            angles_by_step[step_index],
            cvas_by_step[step_index],
            distances_by_step[step_index],
        ):
            return None, None, None

    if not any(angles_by_step):
        return None, None, None

    return angles_by_step, cvas_by_step, distances_by_step


LOOK_STRAIGHT_STEP_INDEX = 0


def run_calibration(cap, landmarker, calib_window):
    """
    Runs a two-phase calibration (good posture, then slouched posture) and
    returns (threshold, cva_threshold_deg, distance_threshold_px, debug_stats),
    where debug_stats is a dict of the raw means/stds behind those thresholds
    (for reference only, not read back by load_calibration_settings). Returns
    (None, None, None, None) if either phase was aborted, the window was
    closed, or produced no data. All instructions/results are shown in
    `calib_window`, never on the console or the webcam feed.
    """
    upright_steps = [
        "Please sit upright, look straight ahead",
    ]
    slouch_steps = [
        "Please slouch",
    ]

    started = calib_window.wait_for_start(
        "Phase 1/2: GOOD POSTURE\n\nGet ready to sit upright and look ahead.\n"
        "Each step will beep, count down to get ready, then beep again and record."
    )
    if not started:
        return None, None, None, None

    calib_window.clear_buttons()
    upright_angles_by_step, upright_cvas_by_step, upright_distances_by_step = run_calibration_phase(
        cap, landmarker, calib_window, "GOOD POSTURE - move naturally", upright_steps, UPRIGHT_STEP_RECORDING_SEC
    )

    if upright_angles_by_step is None:
        try:
            cv2.destroyWindow(CALIBRATION_WINDOW_NAME)
        except cv2.error:
            pass
        if not calib_window.closed:
            calib_window.set_message("Calibration cancelled or no person detected.\nKeeping current settings.")
            calib_window.set_countdown("")
        return None, None, None, None

    started = calib_window.wait_for_start("Phase 2/2: SLOUCHED POSTURE\n\nGet ready to slouch.")
    if not started:
        try:
            cv2.destroyWindow(CALIBRATION_WINDOW_NAME)
        except cv2.error:
            pass
        return None, None, None, None

    calib_window.clear_buttons()
    slouch_angles_by_step, slouch_cvas_by_step, slouch_distances_by_step = run_calibration_phase(
        cap, landmarker, calib_window, "SLOUCHED POSTURE - move naturally", slouch_steps, SLOUCH_STEP_RECORDING_SEC
    )

    try:
        cv2.destroyWindow(CALIBRATION_WINDOW_NAME)
    except cv2.error:
        pass

    if slouch_angles_by_step is None:
        if not calib_window.closed:
            calib_window.set_message("Calibration cancelled or no person detected.\nKeeping current settings.")
            calib_window.set_countdown("")
        return None, None, None, None

    # Only the "look straight ahead" step feeds the vertex-angle calibration.
    upright_look_straight_angles = upright_angles_by_step[LOOK_STRAIGHT_STEP_INDEX]
    slouch_all_angles = [a for step_angles in slouch_angles_by_step for a in step_angles]

    if not upright_look_straight_angles or not slouch_all_angles:
        if not calib_window.closed:
            calib_window.set_message("Calibration cancelled or no person detected.\nKeeping current settings.")
            calib_window.set_countdown("")
        return None, None, None, None

    upright_mean = statistics.mean(upright_look_straight_angles)
    upright_std = statistics.pstdev(upright_look_straight_angles)
    slouch_mean = statistics.mean(slouch_all_angles)
    slouch_std = statistics.pstdev(slouch_all_angles)

    threshold = ((slouch_mean - slouch_std) + (upright_mean + upright_std)) / 2
    # threshold = (slouch_mean + upright_mean) / 2

    # Only the "look straight ahead" step is used for the upright CVA baseline -
    # turning the head left/right shifts the nose sideways relative to the neck
    # point for reasons unrelated to slouching, which would otherwise contaminate it.
    upright_look_straight_cvas = upright_cvas_by_step[LOOK_STRAIGHT_STEP_INDEX]
    slouch_all_cvas = [c for step_cvas in slouch_cvas_by_step for c in step_cvas]

    if upright_look_straight_cvas and slouch_all_cvas:
        upright_cva_mean = statistics.mean(upright_look_straight_cvas)
        upright_cva_std = statistics.pstdev(upright_look_straight_cvas)
        slouch_cva_mean = statistics.mean(slouch_all_cvas)
        slouch_cva_std = statistics.pstdev(slouch_all_cvas)
        cva_threshold_deg = (upright_cva_mean + slouch_cva_mean) / 2
        cva_note = ""
    else:
        upright_cva_mean = upright_cva_std = None
        slouch_cva_mean = slouch_cva_std = None
        cva_threshold_deg = CVA_ANGLE_ALERT_THRESHOLD_DEG
        cva_note = "\n(Not enough data for CVA - kept current CVA threshold.)"

    # Same "look straight ahead" / all-slouch-steps split as CVA above. Nose-neck
    # distance increases when slouching (same direction as the vertex angle), so
    # the threshold is the plain midpoint of the two means, matching the CVA formula.
    upright_look_straight_distances = upright_distances_by_step[LOOK_STRAIGHT_STEP_INDEX]
    slouch_all_distances = [d for step_distances in slouch_distances_by_step for d in step_distances]

    if upright_look_straight_distances and slouch_all_distances:
        upright_distance_mean = statistics.mean(upright_look_straight_distances)
        upright_distance_std = statistics.pstdev(upright_look_straight_distances)
        slouch_distance_mean = statistics.mean(slouch_all_distances)
        slouch_distance_std = statistics.pstdev(slouch_all_distances)
        distance_threshold_px = (upright_distance_mean + slouch_distance_mean) / 2
        distance_note = ""
    else:
        upright_distance_mean = upright_distance_std = None
        slouch_distance_mean = slouch_distance_std = None
        distance_threshold_px = NOSE_NECK_DISTANCE_ALERT_THRESHOLD_PX
        distance_note = "\n(Not enough data for distance - kept current distance threshold.)"

    calib_window.set_message(
        "Calibration results:\n\n"
        f"Good posture:  mean={upright_mean:.1f} deg, std={upright_std:.1f} deg\n"
        f"Slouched:      mean={slouch_mean:.1f} deg, std={slouch_std:.1f} deg\n\n"
        f"New alert threshold: {threshold:.1f} deg\n"
        f"New craniovertebral angle threshold: {cva_threshold_deg:.1f} deg"
        f"{cva_note}\n"
        f"New nose-neck distance threshold: {distance_threshold_px:.1f} px"
        f"{distance_note}"
    )
    calib_window.set_countdown("")

    debug_stats = {
        "upright_angle_mean": upright_mean,
        "upright_angle_std": upright_std,
        "slouch_angle_mean": slouch_mean,
        "slouch_angle_std": slouch_std,
        "upright_cva_mean": upright_cva_mean,
        "upright_cva_std": upright_cva_std,
        "slouch_cva_mean": slouch_cva_mean,
        "slouch_cva_std": slouch_cva_std,
        "upright_distance_mean": upright_distance_mean,
        "upright_distance_std": upright_distance_std,
        "slouch_distance_mean": slouch_distance_mean,
        "slouch_distance_std": slouch_distance_std,
    }

    return threshold, cva_threshold_deg, distance_threshold_px, debug_stats

pose_estimator = Body(mode="lightweight", backend="onnxruntime", device="cpu")

# rtmlib's Body has no context-manager protocol (no cleanup needed, unlike
# MediaPipe's PoseLandmarker), so this just passes it through unchanged -
# kept as a `with` purely to avoid re-indenting the rest of the file.
with contextlib.nullcontext(pose_estimator) as landmarker:
    calib_window = CalibrationWindow()
    try:
        is_camera_above = calib_window.ask_yes_no(
            "When you're sat normally, is the camera positioned above the midpoint in front of you, or below it?",
            yes_text="Above",
            no_text="Below",
        )
        # USE_NECK_VERTEX_ANGLE also drives which threshold formula the
        # non-ML fallback below uses - it's the same underlying answer, just
        # under the name the original threshold logic already expects.
        USE_NECK_VERTEX_ANGLE = is_camera_above

        position = "above" if is_camera_above else "below"
        available_models = discover_available_models(position)

        slouch_model_path = None
        chosen_model_name = None
        if available_models and not calib_window.closed:
            chosen_model_name = calib_window.ask_choice(
                "Select which trained model to use:",
                [(name, MODEL_TYPE_LABELS.get(name, name)) for name in available_models],
            )
            if chosen_model_name is not None:
                slouch_model_path = available_models[chosen_model_name]

        if slouch_model_path is None:
            slouch_model_path = resolve_slouch_model_path(is_camera_above)
            chosen_model_name = None  # legacy fallback path isn't one of the fixed MODEL_TYPE_LABELS entries
        slouch_model = load_slouch_model(slouch_model_path)
        current_model_name = chosen_model_name if slouch_model is not None else None

        if slouch_model is not None:
            print(f"Loaded trained slouch model from {slouch_model_path}; skipping threshold calibration.")
        else:
            calibrate_choice = not calib_window.closed and calib_window.ask_yes_no(
                "Would you like to calibrate posture thresholds for this session?"
            )
            if calibrate_choice and check_volume_and_warn(calib_window):
                calibrated_threshold, calibrated_cva_threshold, calibrated_distance_threshold, calibration_debug_stats = run_calibration(
                    cap, landmarker, calib_window
                )
                if calibrated_threshold is not None:
                    NECK_VERTEX_ALERT_THRESHOLD_DEG = calibrated_threshold
                    CVA_ANGLE_ALERT_THRESHOLD_DEG = calibrated_cva_threshold
                    NOSE_NECK_DISTANCE_ALERT_THRESHOLD_PX = calibrated_distance_threshold
                    if not calib_window.closed:
                        save_choice = calib_window.ask_yes_no("Save these as your permanent posture settings?")
                        if save_choice:
                            save_calibration_settings(
                                CALIBRATION_FILE,
                                calibrated_threshold,
                                calibrated_cva_threshold,
                                calibrated_distance_threshold,
                                calibration_debug_stats,
                            )
    finally:
        # Hidden, not destroyed - kept alive as the one shared Tk root for
        # the whole session, reused later by the switch-model dialog and
        # posture-alert popups instead of spinning up a second Tk() root.
        calib_window.hide()
        # From here on, closing this window (e.g. dismissing a mid-session
        # switch-model prompt) should just cancel that one prompt and keep
        # the current model, not exit the whole app the way closing it
        # during the startup dialogs above does.
        calib_window.exit_on_close = False

    if calib_window.closed_by_user:
        sys.exit(0)

    cv2.namedWindow(VIDEO_WINDOW_NAME, cv2.WINDOW_NORMAL)
    set_window_icon(VIDEO_WINDOW_NAME, APP_ICON_PATH)

    # Hit-box for the "Switch Model" button, recomputed each frame (drawn
    # top-right, sized to the frame's current width) so on_video_mouse always
    # tests clicks against where the button was actually drawn. Only clickable
    # when there's more than one trained model type to switch between.
    switch_model_button = {"x1": 0, "y1": 0, "x2": 0, "y2": 0, "visible": False, "clicked": False}

    def on_video_mouse(event, x, y, flags, userdata):
        if (
            event == cv2.EVENT_LBUTTONDOWN
            and switch_model_button["visible"]
            and switch_model_button["x1"] <= x <= switch_model_button["x2"]
            and switch_model_button["y1"] <= y <= switch_model_button["y2"]
        ):
            switch_model_button["clicked"] = True

    cv2.setMouseCallback(VIDEO_WINDOW_NAME, on_video_mouse)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame.")
            break

        # Drives any active posture-alert Toplevel (rendering it and running
        # its after()-scheduled auto-dismiss) - harmless no-op the rest of
        # the time, since calib_window itself stays hidden between prompts.
        calib_window.pump()

        frame_height = frame.shape[0]

        keypoints, scores = landmarker(frame)

        # Draw full pose skeleton on the frame
        draw_skeleton(frame, keypoints, scores)

        selected = extract_landmarks(keypoints, scores)

        if is_person_present(selected):
            last_person_seen_time = time.time()
            monitoring_enabled = True
        elif time.time() - last_person_seen_time >= NO_PERSON_DISABLE_TIMEOUT_SEC:
            monitoring_enabled = False

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

            if slouch_model is not None:
                shoulder_tilt_angle = angle_from_horizontal(selected["left_shoulder"], selected["right_shoulder"])
                ear_tilt_angle = angle_from_horizontal(selected["left_ear"], selected["right_ear"])
                feature_row = compute_ml_feature_row(
                    selected, vertex_angle, cva_angle, shoulder_tilt_angle, ear_tilt_angle, is_camera_above
                )
                is_bad_posture_frame = (
                    monitoring_enabled and feature_row is not None and predict_slouch(slouch_model, feature_row)
                )
            else:
                is_bad_posture_frame = monitoring_enabled and (
                    (
                        USE_NECK_VERTEX_ANGLE
                        and vertex_angle is not None
                        and vertex_angle > NECK_VERTEX_ALERT_THRESHOLD_DEG
                    )
                    or (
                        not USE_NECK_VERTEX_ANGLE
                        and cva_angle is not None
                        and cva_angle < CVA_ANGLE_ALERT_THRESHOLD_DEG
                        and nose_neck_distance >= NOSE_NECK_DISTANCE_ALERT_THRESHOLD_PX
                    )
                )

            now = time.time()
            if is_bad_posture_frame:
                good_posture_since = None
                if bad_posture_since is None:
                    bad_posture_since = now
                elif (
                    now - bad_posture_since >= SUSTAINED_BAD_POSTURE_SEC
                    and now - last_posture_alert_time > NECK_VERTEX_ALERT_COOLDOWN_SEC
                ):
                    alert_bad_posture(calib_window, vertex_angle)
                    last_posture_alert_time = now
            else:
                # A single noisy frame reading as "fine" shouldn't wipe out an
                # otherwise-real sustained slouch, so bad_posture_since only
                # clears once good posture has held continuously for a short
                # grace period, not on the very next frame.
                if good_posture_since is None:
                    good_posture_since = now
                elif now - good_posture_since >= GOOD_POSTURE_GRACE_SEC:
                    bad_posture_since = None

            cv2.line(frame, (selected["left_shoulder"]["px"], selected["left_shoulder"]["py"]), (neck["px"], neck["py"]), (0, 200, 255), 2)
            cv2.line(frame, (neck["px"], neck["py"]), (selected["right_shoulder"]["px"], selected["right_shoulder"]["py"]), (0, 200, 255), 2)
            cv2.line(frame, (neck["px"], neck["py"]), (nose["px"], nose["py"]), (255, 100, 0), 2)

            # Show all angles stacked in the top-left corner
            overlay_lines = []
            if slouch_model is not None:
                model_label = MODEL_TYPE_LABELS.get(current_model_name, current_model_name or "unknown")
                overlay_lines.append(f"posture detection: ML model ({model_label})")
            else:
                overlay_lines.append("posture detection: calibrated thresholds")
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

            now = time.time()
            if now - last_print_time > 5.0:
                print("slouch" if is_bad_posture_frame else "no slouch")
                last_print_time = now

        if not monitoring_enabled:
            cv2.putText(
                frame,
                "Posture monitoring paused - no person detected",
                (10, frame_height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        if available_models:
            pad_x, pad_y = 10, 8
            (text_w, text_h), _ = cv2.getTextSize(SWITCH_MODEL_BUTTON_LABEL, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            x2 = frame.shape[1] - 10
            x1 = x2 - text_w - 2 * pad_x
            y1 = 10
            y2 = y1 + text_h + 2 * pad_y
            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), 1)
            cv2.putText(
                frame,
                SWITCH_MODEL_BUTTON_LABEL,
                (x1 + pad_x, y2 - pad_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            switch_model_button.update(x1=x1, y1=y1, x2=x2, y2=y2, visible=True)
        else:
            switch_model_button["visible"] = False

        cv2.imshow(VIDEO_WINDOW_NAME, frame)

        if switch_model_button["clicked"]:
            switch_model_button["clicked"] = False
            result = prompt_model_switch(calib_window, available_models)
            if result is not None:
                current_model_name, slouch_model = result
                bad_posture_since = None
                good_posture_since = None
                if slouch_model is not None:
                    print(f"Switched posture detection to {MODEL_TYPE_LABELS.get(current_model_name, current_model_name)} model.")
                else:
                    print("Switched posture detection to calibrated thresholds.")

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Closing the window via the OS close button can fully tear down the
        # underlying window immediately on some backends (observed with
        # OpenCV's Qt backend on Linux), so querying a now-gone window raises
        # cv2.error instead of just reporting it as not visible the way the
        # GTK/Win32 backends do. Either outcome means the same thing here.
        try:
            if cv2.getWindowProperty(VIDEO_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

cap.release()
cv2.destroyAllWindows()
calib_window.close()
