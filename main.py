import ctypes
import json
import math
import os
import random
import statistics
import sys
import threading
import time
import tkinter as tk
import urllib.request
import winsound

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from pycaw.pycaw import AudioUtilities
from pygrabber.dshow_graph import FilterGraph


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


MODEL_PATH = resource_path("pose_landmarker_lite.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
CALIBRATION_ICON_PATH = resource_path("images", "SlouchImageopt.png")
APP_ICON_PATH = resource_path("images", "SlouchLess.ico")
VIDEO_WINDOW_NAME = "SlouchLess"
# default is 123.
NECK_VERTEX_ALERT_THRESHOLD_DEG = 123
NECK_VERTEX_ALERT_COOLDOWN_SEC = 15
SUSTAINED_BAD_POSTURE_SEC = 2
GOOD_POSTURE_GRACE_SEC = 1
MIN_NOSE_NECK_DISTANCE_PX = 35
CVA_ANGLE_ALERT_THRESHOLD_DEG = 85.0
USE_NECK_VERTEX_ANGLE = True
NO_PERSON_DISABLE_TIMEOUT_SEC = 30
MIN_PERSON_PRESENCE_VISIBILITY = 0.6

CALIBRATION_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "SlouchLess")
os.makedirs(CALIBRATION_DIR, exist_ok=True)
CALIBRATION_FILE = os.path.join(CALIBRATION_DIR, "calibration.json")
UPRIGHT_STEP_RECORDING_SEC = 5
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
    "That posture is not giving main character energy.",
    "Sit up before you turn into a question mark.",
]


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
        volume = AudioUtilities.GetSpeakers().EndpointVolume
        is_muted = bool(volume.GetMute())
        volume_level = volume.GetMasterVolumeLevelScalar()
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


def is_person_present(selected):
    """
    In VIDEO mode MediaPipe tracks the previous detection instead of re-running
    full detection every frame, so it can lock onto a static background object
    (a chair, a coat on a hook) and keep reporting it as a person indefinitely.
    Requiring decent visibility on the core landmarks filters most of those out,
    since a real person's shoulders/nose are confidently visible while a
    misidentified object's landmarks tend to be low-confidence guesses.
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


def draw_skeleton(frame, result):
    if not result.pose_landmarks:
        return

    landmarks = result.pose_landmarks[0]
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for start, end in POSE_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 255, 255), 1, cv2.LINE_AA)


def load_calibration_settings(path, default_threshold, default_min_distance, default_cva_threshold):
    """
    Loads previously saved calibration values from disk. Each value falls back
    independently to its default if the file is missing, unparsable, or only
    has some of the keys.
    """
    if not os.path.exists(path):
        return default_threshold, default_min_distance, default_cva_threshold

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        print(f"Could not read calibration file at {path}, using default values.")
        return default_threshold, default_min_distance, default_cva_threshold

    try:
        threshold = float(data["neck_vertex_alert_threshold_deg"])
    except (KeyError, ValueError):
        threshold = default_threshold

    try:
        min_distance = float(data["min_nose_neck_distance_px"])
    except (KeyError, ValueError):
        min_distance = default_min_distance

    try:
        cva_threshold = float(data["cva_angle_alert_threshold_deg"])
    except (KeyError, ValueError):
        cva_threshold = default_cva_threshold

    return threshold, min_distance, cva_threshold


def save_calibration_settings(path, threshold, min_distance, cva_threshold, debug_stats=None):
    """
    `debug_stats`, if given, is written under a separate key purely for the
    user's own reference - it's never read back by load_calibration_settings.
    """
    data = {
        "neck_vertex_alert_threshold_deg": threshold,
        "min_nose_neck_distance_px": min_distance,
        "cva_angle_alert_threshold_deg": cva_threshold,
    }
    if debug_stats is not None:
        data["calibration_debug_stats"] = debug_stats

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(
        f"Saved calibrated threshold ({threshold:.1f} deg), min nose-neck distance ({min_distance:.1f} px), "
        f"and CVA angle threshold ({cva_threshold:.1f} deg) to {path}"
    )


def ensure_model_downloaded(path, url):
    """
    Downloads the MediaPipe model file if it isn't already present locally.
    Streams to a .part file first and renames on success, so an interrupted
    download can't leave a corrupt model file at the real path.
    """
    if os.path.exists(path):
        return

    print(f"Model not found at {path}, downloading from {url} ...")
    tmp_path = path + ".part"
    try:
        urllib.request.urlretrieve(url, tmp_path)
        os.replace(tmp_path, path)
        print("Model download complete.")
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


ensure_model_downloaded(MODEL_PATH, MODEL_URL)

NECK_VERTEX_ALERT_THRESHOLD_DEG, MIN_NOSE_NECK_DISTANCE_PX, CVA_ANGLE_ALERT_THRESHOLD_DEG = load_calibration_settings(
    CALIBRATION_FILE, NECK_VERTEX_ALERT_THRESHOLD_DEG, MIN_NOSE_NECK_DISTANCE_PX, CVA_ANGLE_ALERT_THRESHOLD_DEG
)

options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def list_camera_names():
    """
    Returns detected camera friendly names in the same order DirectShow
    assigns indices, so camera_names[i] corresponds to VideoCapture(i,
    cv2.CAP_DSHOW). Returns [] if enumeration fails for any reason.
    """
    try:
        return FilterGraph().get_input_devices()
    except Exception:
        return []


def prompt_for_camera(camera_names):
    """
    Shows a window listing detected camera names as buttons and blocks until
    one is clicked. Returns the chosen index, or 0 if the window is closed
    without a selection.
    """
    root = tk.Tk()
    root.title("SlouchLess - Select Camera")
    root.attributes("-topmost", True)

    try:
        icon_image = tk.PhotoImage(file=CALIBRATION_ICON_PATH)
        root.iconphoto(True, icon_image)
    except tk.TclError:
        pass

    chosen = {"index": 0}

    def choose(index):
        chosen["index"] = index
        root.destroy()

    tk.Label(root, text="Select which camera to use:", font=("Segoe UI", 12)).pack(padx=20, pady=(20, 10))
    for index, name in enumerate(camera_names):
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


camera_names = list_camera_names()
camera_index = prompt_for_camera(camera_names) if camera_names else 0
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Could not open the selected webcam.")

last_print_time = time.time()
last_posture_alert_time = 0.0
last_person_seen_time = time.time()
monitoring_enabled = True
bad_posture_since = None
good_posture_since = None


def _show_posture_popup(message):
    root = tk.Tk()
    root.title("Posture Alert")
    root.attributes("-topmost", True)
    root.overrideredirect(True)
    root.configure(bg="#1e1e1e")

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

    # Size the window to fit the message instead of a fixed geometry, so a
    # longer BAD_POSTURE_MESSAGES entry that wraps to more lines doesn't get
    # clipped by a too-short fixed height.
    root.update_idletasks()
    width = root.winfo_reqwidth()
    height = root.winfo_reqheight()
    x = (root.winfo_screenwidth() - width) // 2
    y = (root.winfo_screenheight() - height) // 2
    root.geometry(f"{width}x{height}+{x}+{y}")

    root.after(4000, root.destroy)
    root.mainloop()


def alert_bad_posture(vertex_angle):
    message = random.choice(BAD_POSTURE_MESSAGES)
    threading.Thread(target=_show_posture_popup, args=(message,), daemon=True).start()


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
        self._destroyed = False

    def _on_close(self):
        self._response = False
        self.close()

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
        while self._response is None and not self.closed:
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


def _play_beep():
    winsound.Beep(BEEP_FREQUENCY_HZ, BEEP_DURATION_MS)


def _read_and_detect(cap, landmarker):
    """
    Reads one frame, runs pose detection, draws the skeleton, and extracts the
    posture landmarks. Returns (frame, selected), or (None, None) if the camera
    read failed.
    """
    ret, frame = cap.read()
    if not ret:
        return None, None

    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(time.time() * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    draw_skeleton(frame, result)
    selected = extract_landmarks(result, frame_width, frame_height)

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


def _run_calibration_recording(cap, landmarker, calib_window, phase_title, instruction, duration_sec, angles, distances, cvas):
    """
    Shows `instruction` as the active step with a countdown in `calib_window`,
    appending the neck vertex angle, nose-neck distance, and craniovertebral
    angle of every confidently detected frame to `angles`/`distances`/`cvas`.
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
            distances.append(math.hypot(nose["px"] - neck["px"], nose["py"] - neck["py"]))

            cva_angle = angle_from_horizontal(neck, nose)
            if cva_angle is not None:
                cvas.append(cva_angle)

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
    beep again and record the neck vertex angle and nose-neck distance for
    step_duration_sec seconds. `step_duration_sec` is a required, explicit
    per-step length rather than a total split across len(steps), so changing
    how many steps a phase has doesn't silently change each step's length.

    Returns (angles_by_step, cvas_by_step, nose_neck_distances_by_step),
    each a list of per-frame lists in the same order as `steps`. Returns
    (None, None, None) if any step was aborted with 'q', the window was
    closed, or no person was ever detected. Reuses the already-open
    `cap`/`landmarker`.
    """
    angles_by_step = [[] for _ in steps]
    cvas_by_step = [[] for _ in steps]
    nose_neck_distances_by_step = [[] for _ in steps]

    for step_index, instruction in enumerate(steps):
        _play_beep()
        if not _run_calibration_countdown(cap, landmarker, calib_window, phase_title, instruction, CALIBRATION_PREP_COUNTDOWN_SEC):
            return None, None, None

        _play_beep()
        if not _run_calibration_recording(
            cap,
            landmarker,
            calib_window,
            phase_title,
            instruction,
            step_duration_sec,
            angles_by_step[step_index],
            nose_neck_distances_by_step[step_index],
            cvas_by_step[step_index],
        ):
            return None, None, None

    if not any(angles_by_step):
        return None, None, None

    return angles_by_step, cvas_by_step, nose_neck_distances_by_step


LOOK_STRAIGHT_STEP_INDEX = 0
LOOK_DOWN_STEP_INDEX = 1


def run_calibration(cap, landmarker, calib_window):
    """
    Runs a two-phase calibration (good posture, then slouched posture) and
    returns (threshold, min_nose_neck_distance, cva_threshold_deg, debug_stats),
    where debug_stats is a dict of the raw means/stds behind those thresholds
    (for reference only, not read back by load_calibration_settings). Returns
    (None, None, None, None) if either phase was aborted, the window was
    closed, or produced no data. All instructions/results are shown in
    `calib_window`, never on the console or the webcam feed.
    """
    upright_steps = [
        "Sit upright, look straight ahead",
        "Look down slightly (enough to see keyboard and front of you), with upright position",
    ]
    slouch_steps = [
        "Slouch and slightly move your head left and right",
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
    slouch_angles_by_step, slouch_cvas_by_step, _ = run_calibration_phase(
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

    # Only the "look straight ahead" step feeds the vertex-angle calibration -
    # the "look down" step is kept for the nose-neck distance calibration below,
    # but looking down naturally raises the vertex angle for reasons unrelated
    # to slouching, which would otherwise skew the upright baseline.
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
        cva_threshold_deg = ((upright_cva_mean - upright_cva_std) + (slouch_cva_mean + slouch_cva_std)) / 2
        cva_note = ""
    else:
        upright_cva_mean = upright_cva_std = None
        slouch_cva_mean = slouch_cva_std = None
        cva_threshold_deg = CVA_ANGLE_ALERT_THRESHOLD_DEG
        cva_note = "\n(Not enough data for CVA - kept current CVA threshold.)"

    look_down_distances = upright_distances_by_step[LOOK_DOWN_STEP_INDEX]
    if look_down_distances:
        look_down_distance_mean = statistics.mean(look_down_distances)
        look_down_distance_std = statistics.pstdev(look_down_distances)
        min_nose_neck_distance = look_down_distance_mean + look_down_distance_std
        distance_note = ""
    else:
        look_down_distance_mean = look_down_distance_std = None
        min_nose_neck_distance = MIN_NOSE_NECK_DISTANCE_PX
        distance_note = "\n(No person detected during 'look down' - kept current min distance.)"

    calib_window.set_message(
        "Calibration results:\n\n"
        f"Good posture:  mean={upright_mean:.1f} deg, std={upright_std:.1f} deg\n"
        f"Slouched:      mean={slouch_mean:.1f} deg, std={slouch_std:.1f} deg\n\n"
        f"New alert threshold: {threshold:.1f} deg\n"
        f"New min nose-neck distance: {min_nose_neck_distance:.1f} px"
        f"{distance_note}\n"
        f"New craniovertebral angle threshold: {cva_threshold_deg:.1f} deg"
        f"{cva_note}"
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
        "look_down_distance_mean": look_down_distance_mean,
        "look_down_distance_std": look_down_distance_std,
    }

    return threshold, min_nose_neck_distance, cva_threshold_deg, debug_stats

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    calib_window = CalibrationWindow()
    try:
        USE_NECK_VERTEX_ANGLE = calib_window.ask_yes_no(
            "When you're sat normally, is the camera positioned above the midpoint in front of you, or below it?",
            yes_text="Above",
            no_text="Below",
        )

        calibrate_choice = not calib_window.closed and calib_window.ask_yes_no(
            "Would you like to calibrate posture thresholds for this session?"
        )
        if calibrate_choice and check_volume_and_warn(calib_window):
            calibrated_threshold, calibrated_min_distance, calibrated_cva_threshold, calibration_debug_stats = run_calibration(
                cap, landmarker, calib_window
            )
            if calibrated_threshold is not None:
                NECK_VERTEX_ALERT_THRESHOLD_DEG = calibrated_threshold
                MIN_NOSE_NECK_DISTANCE_PX = calibrated_min_distance
                CVA_ANGLE_ALERT_THRESHOLD_DEG = calibrated_cva_threshold
                if not calib_window.closed:
                    save_choice = calib_window.ask_yes_no("Save these as your permanent posture settings?")
                    if save_choice:
                        save_calibration_settings(
                            CALIBRATION_FILE,
                            calibrated_threshold,
                            calibrated_min_distance,
                            calibrated_cva_threshold,
                            calibration_debug_stats,
                        )
    finally:
        calib_window.close()

    cv2.namedWindow(VIDEO_WINDOW_NAME, cv2.WINDOW_NORMAL)
    set_window_icon(VIDEO_WINDOW_NAME, APP_ICON_PATH)

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

            is_bad_posture_frame = monitoring_enabled and (
                nose_neck_distance >= MIN_NOSE_NECK_DISTANCE_PX
            ) and (
                (
                    USE_NECK_VERTEX_ANGLE
                    and vertex_angle is not None
                    and vertex_angle > NECK_VERTEX_ALERT_THRESHOLD_DEG
                )
                or (
                    not USE_NECK_VERTEX_ANGLE
                    and cva_angle is not None
                    and cva_angle < CVA_ANGLE_ALERT_THRESHOLD_DEG
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
                    alert_bad_posture(vertex_angle)
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

        cv2.imshow(VIDEO_WINDOW_NAME, frame)

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if cv2.getWindowProperty(VIDEO_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv2.destroyAllWindows()
