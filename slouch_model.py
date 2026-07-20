import math
import os

import joblib
import pandas as pd

from app_paths import SLOUCH_MODEL_PATH, resource_path

MODEL_TYPE_LABELS = {
    "mlp": "Neural Net (MLP)",
    "svm_rbf": "SVM (RBF kernel)",
    "random_forest": "Random Forest",
}
# ask_choice() returns the chosen key verbatim, so this needs to be a value
# that can never collide with an entry in MODEL_TYPE_LABELS.
THRESHOLDS_CHOICE_KEY = "__calibrated_thresholds__"
SWITCH_MODEL_BUTTON_LABEL = "Switch Model"


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
