"""Single source of truth for the Vex Kernel version.

All version references across the codebase MUST import from here.
Do NOT hardcode version strings in server.py, package.json, etc.

Bump this value and run the sync script (or CI) to propagate.
"""

from typing import Final

VERSION: Final[str] = "2.4.0"
