import json
import os


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
