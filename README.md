# SlouchLess

SlouchLess is a lightweight desktop app that watches your posture through your camera and nudges you when you start slouching.

### How it works
- Uses [MediaPipe](https://developers.google.com/mediapipe) pose detection to track your neck, shoulders, and head position in real time.
- A short guided calibration (sit upright, then slouch on cue) learns *your* personal posture thresholds instead of relying on one-size-fits-all defaults.
- Adapts its detection method to where your webcam actually sits - above your monitor or below it - since that changes which signals are reliable.
- Pops up a (mildly judgmental) reminder once it's confident you've been slouching for some time.

## Just want to run it?

Download the prebuilt binary for your OS from the [Releases page](https://github.com/thisishale/SlouchLess/releases/tag/v1.0.0).

## Running from source

Requirements:
- Python 3.9+ (developed/tested on 3.12)
- A webcam or built-in camera
- **Linux only**: a few system packages not covered by pip:
  ```bash
  sudo apt install python3-tk libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
  ```
  (`python3-tk` for the calibration UI, the rest for OpenCV's display window)

Setup:

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux
source venv/bin/activate

pip install -r requirements.txt
python main.py
```

## Building the exe/binary yourself

After following the "Running from source" setup above, just add PyInstaller and build:

```bash
pip install pyinstaller
pyinstaller --noconfirm SlouchLess.spec
```

The build output lands in `dist/`

Notes:
- On Linux, PyInstaller also needs the `binutils` package (`objdump`) installed system-wide: `sudo apt install binutils`.
- Supports two camera positions, above or below the monitor. Results in other setups aren't verified.

## Acknowledgments

Directing architecture and posture-detection logic, debugging cross-platform issues, and tuning calibration by hand. Code built iteratively with [Claude Code](https://claude.com/claude-code)

