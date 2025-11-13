# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

from __future__ import annotations

import sys

from gemseo import _configure_logger
from gemseo.utils.constants import _LOGGING_DATE_FORMAT
from gemseo.utils.constants import _LOGGING_FILE_MODE
from gemseo.utils.constants import _LOGGING_FILE_PATH
from gemseo.utils.constants import _LOGGING_LEVEL
from gemseo.utils.constants import _LOGGING_MESSAGE_FORMAT


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
    _configure_logger(
        "",
        _LOGGING_LEVEL,
        _LOGGING_MESSAGE_FORMAT,
        _LOGGING_DATE_FORMAT,
        _LOGGING_FILE_PATH,
        _LOGGING_FILE_MODE,
        WrapStdOut(),
    )
