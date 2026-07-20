import glob
import tkinter as tk

from app_paths import CALIBRATION_ICON_PATH, IS_WINDOWS

if IS_WINDOWS:
    from pygrabber.dshow_graph import FilterGraph


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
