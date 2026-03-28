"""
Root-level conftest.py - Patches tempfile to allow pytest to run.
"""
import sys
import os

# Find a writable temp directory
_TEMP_DIR = None
for _c in [
    r'D:\310Programm\futureQuant\tests',
]:
    try:
        _m = os.path.join(_c, '.fq_temp_ok.tmp')
        with open(_m, 'w') as _f:
            _f.write('')
        os.remove(_m)
        _TEMP_DIR = _c
        break
    except Exception:
        pass

if _TEMP_DIR is None:
    _TEMP_DIR = r'D:\310Programm\futureQuant\tests'

os.environ['TMP'] = _TEMP_DIR
os.environ['TEMP'] = _TEMP_DIR

# Patch tempfile
import tempfile as _tf

_tf.tempdir = _TEMP_DIR

def _safe_gettempdir():
    return _TEMP_DIR
_tf.gettempdir = _safe_gettempdir

try:
    _tf._gettempdir = _safe_gettempdir
except (TypeError, AttributeError):
    pass
