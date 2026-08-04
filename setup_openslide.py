# -*- coding: utf-8 -*-
"""
FusionDx -- OpenSlide Windows Setup Helper
============================================
Run this script AFTER installing the OpenSlide Windows binaries and adding
their bin/ folder to your PATH.  It verifies the installation and, if needed,
helps you locate the DLLs.

Usage:
    venv\\Scripts\\python.exe setup_openslide.py

If OpenSlide is not yet installed, this script will print clear instructions.
"""

import os
import sys
import ctypes
import subprocess
from pathlib import Path


OPENSLIDE_DOWNLOAD_URL = "https://openslide.org/download/"
OPENSLIDE_DLLS = ["libopenslide-1.dll", "libopenslide-0.dll"]


def check_path_for_openslide():
    """Search PATH directories for the OpenSlide DLL."""
    found_dirs = []
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        p = Path(entry)
        if not p.is_dir():
            continue
        for dll_name in OPENSLIDE_DLLS:
            if (p / dll_name).exists():
                found_dirs.append((str(p), dll_name))
    return found_dirs


def try_load_openslide_dll(directory: str) -> bool:
    """Attempt to load the OpenSlide DLL from a specific directory."""
    for dll_name in OPENSLIDE_DLLS:
        dll_path = Path(directory) / dll_name
        if dll_path.exists():
            try:
                ctypes.CDLL(str(dll_path))
                return True
            except OSError:
                pass
    return False


def main():
    print("=" * 60)
    print("FusionDx -- OpenSlide Installation Checker")
    print("=" * 60)

    # 1. Check if openslide-python is pip-installed
    print("\n[1/3] Checking openslide-python pip package ...")
    try:
        import importlib.util
        spec = importlib.util.find_spec("openslide")
        if spec is None:
            raise ImportError("not found")
        print("      openslide-python: installed")
    except ImportError:
        print("      openslide-python: NOT installed")
        print("      Fix: venv\\Scripts\\pip.exe install openslide-python==1.3.1")
        sys.exit(1)

    # 2. Search PATH for the native DLL
    print("\n[2/3] Searching PATH for OpenSlide DLL ...")
    found = check_path_for_openslide()
    if found:
        for d, dll in found:
            print(f"      Found: {dll}  in  {d}")
    else:
        print("      OpenSlide DLL NOT found in PATH")
        print()
        print("  To fix this:")
        print(f"  1. Download OpenSlide Windows binaries from: {OPENSLIDE_DOWNLOAD_URL}")
        print("  2. Extract the zip to a permanent location, e.g. C:\\OpenSlide\\")
        print("  3. Add the bin\\ subfolder to your system PATH:")
        print("       - Open: System Properties -> Advanced -> Environment Variables")
        print("       - Edit the 'Path' system variable")
        print("       - Add entry: C:\\OpenSlide\\bin  (adjust to your actual path)")
        print("       - Click OK, then RESTART your terminal / IDE")
        print("  4. Re-run this script to verify.")
        sys.exit(1)

    # 3. Import openslide (this loads the DLL)
    print("\n[3/3] Importing openslide ...")
    try:
        import openslide
        print(f"      openslide-python version : {openslide.__version__}")
        print(f"      OpenSlide library version: {openslide.OPENSLIDE_VERSION}")
        print()
        print("=" * 60)
        print("SUCCESS -- OpenSlide is correctly installed!")
        print()
        print("You can now run the full pipeline:")
        print("  python -m src.data_pipeline --verify")
        print("  python -m src.data_pipeline   (downloads real data)")
        print("  python train_all.py            (trains all models)")
        print("=" * 60)
    except Exception as exc:
        print(f"      FAILED to import openslide: {exc}")
        print()
        print("  The DLL was found in PATH but failed to load.")
        print("  This usually means a missing dependency (e.g. Visual C++ runtime).")
        print("  Try installing: Microsoft Visual C++ Redistributable (x64)")
        print("  Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe")
        sys.exit(1)


if __name__ == "__main__":
    main()
