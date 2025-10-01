"""Reset logging.

Workaround found here: https://github.com/sphinx-gallery/sphinx-gallery/issues/1112
and here: https://github.com/mne-tools/mne-python/blob/6ccd1123ed10cdda2a34d0d4104fbad8f9bae23f/mne/utils/_logging.py#L337-L350
"""

from __future__ import annotations

import logging
import sys


class WrapStdOut:
    """Dynamically wrap to sys.stdout.

    This makes packages that monkey-patch sys.stdout (e.g.doctest,
    sphinx-gallery) work properly.
    """

    def __getattr__(self, name):  # noqa: D105
        # Even more ridiculous than this class, this must be sys.stdout (not
        # just stdout) in order for this to work (tested on OSX and Linux)
        if hasattr(sys.stdout, name):
            return getattr(sys.stdout, name)
        msg = f"'file' object has not attribute '{name}'"
        raise AttributeError(msg)


def reset_logging(gallery_conf, fname):
    root = logging.getLogger()
    for handler in root.handlers:
        root.removeHandler(handler)

    handler = logging.StreamHandler(WrapStdOut())
    root.addHandler(handler)
