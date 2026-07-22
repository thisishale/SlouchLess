# -*- mode: python ; coding: utf-8 -*-
import glob

from PyInstaller.utils.hooks import collect_submodules

datas = [
    ('images/SlouchImageopt.png', 'images'),
    ('images/SlouchLess.ico', 'images'),
] + [(model_file, 'models') for model_file in glob.glob('models/*.joblib')]
binaries = []
# joblib.load() unpickles trained models by dynamically importing internal
# sklearn and numpy submodules (e.g. sklearn.neural_network._multilayer_
# perceptron, numpy._core) that are only referenced inside the pickled data,
# not as literal imports in this project's own source - PyInstaller's static
# analysis misses them, so joblib.load() silently throws in the frozen exe
# and load_slouch_model() falls back to calibrated thresholds no matter
# which model was picked ("No module named 'sklearn...'" / "'numpy._core'").
hiddenimports = collect_submodules('sklearn') + collect_submodules('numpy')


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='SlouchLess',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='images/SlouchLess.ico',
)
