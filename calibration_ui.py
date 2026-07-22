import random
import re
import subprocess
import time
import tkinter as tk

from app_paths import CALIBRATION_ICON_PATH, IS_WINDOWS

if IS_WINDOWS:
    import winsound
    from pycaw.pycaw import AudioUtilities

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
