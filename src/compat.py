"""Platform compatibility shims for WorldForge.

Handles Windows Defender Application Control (WDAC) / AppLocker restrictions
where MuJoCo's optional bundled plugin DLLs throw WinError 4551 on import.
"""

from __future__ import annotations

import ctypes
import sys


def patch_mujoco_windows_dll_loader() -> None:
    """Wrap ctypes.CDLL to safely ignore WDAC-blocked optional MuJoCo plugins."""
    if sys.platform == "win32" and not getattr(ctypes, "_worldforge_cdll_patched", False):
        orig_cdll = ctypes.CDLL

        def safe_cdll(name, *args, **kwargs):
            try:
                return orig_cdll(name, *args, **kwargs)
            except OSError as err:
                # Catch WinError 4551 (Application Control blocked file) or missing DLLs
                # in MuJoCo's optional bundled plugin directory.
                name_str = str(name).lower() if name else ""
                if "mujoco" in name_str and "plugin" in name_str:
                    return None
                raise

        ctypes.CDLL = safe_cdll
        ctypes._worldforge_cdll_patched = True


# Automatically apply on import
patch_mujoco_windows_dll_loader()
