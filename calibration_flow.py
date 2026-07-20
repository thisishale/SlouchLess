import math
import statistics
import time

import cv2

from calibration_ui import _play_beep
from pose_math import angle_from_horizontal, draw_skeleton, extract_landmarks, is_person_present, neck_shoulder_angle

UPRIGHT_STEP_RECORDING_SEC = 10
SLOUCH_STEP_RECORDING_SEC = 10
CALIBRATION_WINDOW_NAME = "Posture Calibration"
CALIBRATION_PREP_COUNTDOWN_SEC = 5

LOOK_STRAIGHT_STEP_INDEX = 0


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


def run_calibration(cap, landmarker, calib_window, default_cva_threshold, default_distance_threshold):
    """
    Runs a two-phase calibration (good posture, then slouched posture) and
    returns (threshold, cva_threshold_deg, distance_threshold_px, debug_stats),
    where debug_stats is a dict of the raw means/stds behind those thresholds
    (for reference only, not read back by load_calibration_settings). Returns
    (None, None, None, None) if either phase was aborted, the window was
    closed, or produced no data. All instructions/results are shown in
    `calib_window`, never on the console or the webcam feed.

    `default_cva_threshold`/`default_distance_threshold` are the caller's
    current session values, used as-is when a phase collects no usable
    CVA/distance data to compute a new threshold from.
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
        cva_threshold_deg = default_cva_threshold
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
        distance_threshold_px = default_distance_threshold
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
