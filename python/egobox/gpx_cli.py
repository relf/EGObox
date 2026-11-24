"""Console script entrypoint for the bundled gpx command."""

from __future__ import annotations

import signal
import sys
import threading

from . import egobox as _native


def main() -> int:
    """Run the Rust gpx CLI through the egobox extension module."""
    # The CLI runs in this process without returning to Python: Python would only raise
    # KeyboardInterrupt once it is over (e.g. after a long fit, its model being written).
    # Ctrl-C terminates it at once instead, as the native gpx binary.
    restore = threading.current_thread() is threading.main_thread()
    if restore:
        previous = signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        return int(_native._run_gpx_cli(sys.argv[1:]))
    finally:
        if restore:
            signal.signal(signal.SIGINT, previous)


if __name__ == "__main__":
    raise SystemExit(main())
