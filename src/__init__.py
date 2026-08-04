# -*- coding: utf-8 -*-
# FusionDx source package
#
# OpenSlide Windows DLL path helper:
# Set OPENSLIDE_PATH env var to your OpenSlide bin/ directory if not in PATH.
# Default tried: C:\OpenSlide\bin

import os
import sys

_openslide_path = os.environ.get("OPENSLIDE_PATH", r"C:\OpenSlide\bin")
if os.path.isdir(_openslide_path) and sys.platform == "win32":
    try:
        os.add_dll_directory(_openslide_path)
    except Exception:
        pass
