import os, sys

from pathlib import Path

import matplotlib.pyplot as plt

# Every plot that a run makes ends with finish_figure(). By default the figure is shown, as
# plt.show() would. run.py configures it to also save the figures in the run's folder,
# and to not show them. Without a display (e.g. a cluster node), figures are never shown.

def has_display() -> bool:
    if os.environ.get("MPLBACKEND", "").lower() == "agg":
        return False
    if sys.platform.startswith("linux"):  # Windows and macOS always have one
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True

_directory = None
_show = has_display()
_count = 0

def configure(directory = None, show = True) -> None:
    global _directory, _show, _count
    _directory = None if directory is None else Path(directory)
    _show = show and has_display()
    _count = 0

def finish_figure(name) -> None:  # Saves the current figure as <directory>/<nn>_<name>.png if configured, then shows or closes it
    global _count
    if _directory is not None:
        _count += 1
        plt.savefig(_directory / f'{_count:02}_{name}.png', bbox_inches="tight")
    if _show:
        plt.show()
    else:
        plt.close("all")
