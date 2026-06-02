"""Console script entrypoint for the bundled gpx command."""

from __future__ import annotations

import sys

from . import egobox as _native


def main() -> int:
    """Run the Rust gpx CLI through the egobox extension module."""
    return int(_native._run_gpx_cli(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
