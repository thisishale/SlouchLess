import sys
import threading
import tkinter as tk
import traceback
from tkinter import messagebox


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


def install_excepthook():
    """
    Routes uncaught exceptions on the main thread and any other thread into
    the same fatal-error dialog, instead of just vanishing (console=False
    build) or silently killing a background thread.
    """
    sys.excepthook = _handle_uncaught_exception
    threading.excepthook = _handle_uncaught_thread_exception
