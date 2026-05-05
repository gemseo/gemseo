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
import warnings
from logging import root

from gemseo import _configure_logger
from gemseo.utils.constants import _LOGGING_DATE_FORMAT
from gemseo.utils.constants import _LOGGING_FILE_MODE
from gemseo.utils.constants import _LOGGING_FILE_PATH
from gemseo.utils.constants import _LOGGING_LEVEL
from gemseo.utils.constants import _LOGGING_MESSAGE_FORMAT
from gemseo.utils.logging import _is_gemseo_logger

# Suppress deprecation warnings as early as possible: mkdocs-gallery parses
# every example with ast.Str (deprecated) inside split_code_and_text_blocks,
# which runs before reset_logging is invoked.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


class _WrapStdOut:
    """Dynamically wrap to sys.stdout.

    This makes packages that monkey-patch sys.stdout (e.g.doctest,
    sphinx-gallery) work properly.
    """

    def __getattr__(self, name):  # noqa: D105
        # Dispatch to the original stdout.
        return getattr(sys.stdout, name)


def reset_logging(gallery_conf, fname):
    # Route every GEMSEO logger through _WrapStdOut so log records reach the
    # current sys.stdout (mkdocs-gallery's _LoggingTee during example
    # execution). The tee captures them for the gallery "Out:" block and
    # forwards them at debug level, keeping the console clean.
    stream = _WrapStdOut()
    for name in root.manager.loggerDict:
        if _is_gemseo_logger(name):
            _configure_logger(
                name,
                _LOGGING_LEVEL,
                _LOGGING_MESSAGE_FORMAT,
                _LOGGING_DATE_FORMAT,
                _LOGGING_FILE_PATH,
                _LOGGING_FILE_MODE,
                stream,
            )
    # Reapply the deprecation filters in case an example or another plugin
    # reset warnings.filters between examples.
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
