import ctypes
import os
import platform
import sys

IS_WINDOWS = platform.system() == "Windows"


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

if IS_WINDOWS:
    CALIBRATION_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "SlouchLess")
else:
    CALIBRATION_DIR = os.path.join(os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")), "SlouchLess")
os.makedirs(CALIBRATION_DIR, exist_ok=True)
CALIBRATION_FILE = os.path.join(CALIBRATION_DIR, "calibration.json")
