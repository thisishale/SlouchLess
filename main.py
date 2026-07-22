import contextlib
import math
import sys
import time
from tkinter import messagebox

import cv2
from rtmlib import Body

from app_paths import APP_ICON_PATH, IS_WINDOWS, CALIBRATION_FILE, set_window_icon
from calibration_flow import run_calibration
from calibration_settings import load_calibration_settings, save_calibration_settings
from calibration_ui import CalibrationWindow, alert_bad_posture, check_volume_and_warn
from camera import list_camera_names, prompt_for_camera
from errors import install_excepthook
from pose_math import angle_from_horizontal, extract_landmarks, is_person_present, neck_shoulder_angle, draw_skeleton
from slouch_model import (
    MODEL_TYPE_LABELS,
    SWITCH_MODEL_BUTTON_LABEL,
    THRESHOLDS_CHOICE_KEY,
    compute_ml_feature_row,
    discover_available_models,
    load_slouch_model,
    predict_slouch,
    prompt_model_switch,
    resolve_slouch_model_path,
)

install_excepthook()

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

NECK_VERTEX_ALERT_THRESHOLD_DEG, CVA_ANGLE_ALERT_THRESHOLD_DEG, NOSE_NECK_DISTANCE_ALERT_THRESHOLD_PX = load_calibration_settings(
    CALIBRATION_FILE, NECK_VERTEX_ALERT_THRESHOLD_DEG, CVA_ANGLE_ALERT_THRESHOLD_DEG, NOSE_NECK_DISTANCE_ALERT_THRESHOLD_PX
)

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
        wants_calibrated_thresholds = False
        if available_models and not calib_window.closed:
            choices = [(name, MODEL_TYPE_LABELS.get(name, name)) for name in MODEL_TYPE_LABELS if name in available_models]
            choices.append((THRESHOLDS_CHOICE_KEY, "Calibrated thresholds"))
            picked = calib_window.ask_choice("Select which trained model to use:", choices)
            if picked == THRESHOLDS_CHOICE_KEY:
                wants_calibrated_thresholds = True
            elif picked is not None:
                chosen_model_name = picked
                slouch_model_path = available_models[picked]

        if slouch_model_path is None and not wants_calibrated_thresholds:
            slouch_model_path = resolve_slouch_model_path(is_camera_above)
            chosen_model_name = None  # legacy fallback path isn't one of the fixed MODEL_TYPE_LABELS entries
        slouch_model = load_slouch_model(slouch_model_path) if slouch_model_path is not None else None
        current_model_name = chosen_model_name if slouch_model is not None else None

        def maybe_calibrate_thresholds(cva_threshold, distance_threshold):
            """
            Asks whether to calibrate posture thresholds for this session and,
            if so, runs the calibration flow and offers to save the results
            permanently. Returns (threshold, cva_threshold, distance_threshold)
            with the new values, or None if the user declined, aborted, or
            nothing usable was collected - callers should leave the existing
            thresholds untouched in that case. Shared by both the startup flow
            below and the mid-session "Switch Model" -> "Calibrated thresholds"
            path, so recalibrating is offered consistently either way.
            """
            # Reopen defensively: harmless (and a no-op) during the startup
            # flow where calib_window is already shown, but essential for the
            # mid-session "Switch Model" -> "Calibrated thresholds" path,
            # where prompt_model_switch() already hid the window again right
            # after that choice - without this, ask_yes_no() below would wait
            # forever for a click on a window nobody can see, looking exactly
            # like a freeze.
            calib_window.reopen()
            try:
                calibrate_choice = not calib_window.closed and calib_window.ask_yes_no(
                    "Would you like to calibrate posture thresholds for this session?"
                )
                if not (calibrate_choice and check_volume_and_warn(calib_window)):
                    return None

                calibrated_threshold, calibrated_cva_threshold, calibrated_distance_threshold, calibration_debug_stats = run_calibration(
                    cap, landmarker, calib_window, cva_threshold, distance_threshold
                )
                if calibrated_threshold is None:
                    return None

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

                return calibrated_threshold, calibrated_cva_threshold, calibrated_distance_threshold
            finally:
                if not calib_window.closed:
                    calib_window.hide()

        if slouch_model is not None:
            print(f"Loaded trained slouch model from {slouch_model_path}; skipping threshold calibration.")
        else:
            calibration_result = maybe_calibrate_thresholds(CVA_ANGLE_ALERT_THRESHOLD_DEG, NOSE_NECK_DISTANCE_ALERT_THRESHOLD_PX)
            if calibration_result is not None:
                NECK_VERTEX_ALERT_THRESHOLD_DEG, CVA_ANGLE_ALERT_THRESHOLD_DEG, NOSE_NECK_DISTANCE_ALERT_THRESHOLD_PX = calibration_result
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

    # cv2 never centers its own windows - left alone, window managers place
    # it wherever their own default policy says (observed pushed to the
    # right edge under GNOME/Wayland via XWayland, though acceptable by
    # coincidence on Windows). Centered explicitly instead, once the frame
    # size is known, using the same screen-size query the Tk dialogs already
    # use (calib_window.root is still alive, just hidden, at this point).
    video_window_positioned = False

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

        if not video_window_positioned:
            frame_width = frame.shape[1]
            screen_width = calib_window.root.winfo_screenwidth()
            screen_height = calib_window.root.winfo_screenheight()
            x = max(0, (screen_width - frame_width) // 2)
            y = max(0, (screen_height - frame_height) // 2)
            cv2.moveWindow(VIDEO_WINDOW_NAME, x, y)
            video_window_positioned = True

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
                    calibration_result = maybe_calibrate_thresholds(CVA_ANGLE_ALERT_THRESHOLD_DEG, NOSE_NECK_DISTANCE_ALERT_THRESHOLD_PX)
                    if calibration_result is not None:
                        NECK_VERTEX_ALERT_THRESHOLD_DEG, CVA_ANGLE_ALERT_THRESHOLD_DEG, NOSE_NECK_DISTANCE_ALERT_THRESHOLD_PX = calibration_result

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
