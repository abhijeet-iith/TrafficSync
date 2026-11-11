"""Locating SUMO and starting TraCI connections."""
from __future__ import annotations

import itertools
import os
import shutil
from pathlib import Path
from typing import Sequence

__all__ = ["sumo_available", "sumo_binary", "start_traci", "tool_binary"]

_labels = itertools.count()


def _sumo_home() -> Path | None:
    if os.environ.get("SUMO_HOME"):
        return Path(os.environ["SUMO_HOME"])
    try:
        import sumo  # the eclipse-sumo wheel

        return Path(sumo.SUMO_HOME)
    except ImportError:
        return None


def tool_binary(name: str) -> str:
    """Path of a SUMO executable (``sumo``, ``sumo-gui``, ``netconvert``, ...)."""
    home = _sumo_home()
    if home is not None:
        for cand in (home / "bin" / name, home / "bin" / f"{name}.exe"):
            if cand.exists():
                return str(cand)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"SUMO tool {name!r} not found; install with `pip install eclipse-sumo` or set SUMO_HOME")


def sumo_binary(gui: bool = False) -> str:
    return tool_binary("sumo-gui" if gui else "sumo")


def sumo_available() -> bool:
    try:
        import traci  # noqa: F401

        tool_binary("sumo")
        tool_binary("netconvert")
        return True
    except (ImportError, FileNotFoundError):
        return False


GUI_ARGS = ["--start", "true", "--delay", "100", "--quit-on-end", "false"]


def start_traci(args: Sequence[str], gui: bool = False, gui_args: Sequence[str] | None = None):
    """Start SUMO with ``args`` and return a TraCI connection (each call gets its own label).

    TraCI prints " Retrying in 1 seconds" while SUMO is still starting; that message is normal and is
    suppressed here.  ``gui=True`` opens ``sumo-gui`` (playing immediately, 100 ms per step by default)."""
    import contextlib
    import io

    import traci

    label = f"ts{os.getpid()}_{next(_labels)}"
    cmd = [sumo_binary(gui), *args, *((GUI_ARGS if gui_args is None else gui_args) if gui else [])]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            traci.start(cmd, label=label, numRetries=120)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"could not start SUMO ({' '.join(cmd[:3])} ...): {e}\n"
                           "check that `sumo --version` works and that traci/sumolib match the SUMO version") from e
    return traci.getConnection(label)
