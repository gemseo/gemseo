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

"""MkDocs hook restoring the working directory around every build.

The mkdocs-gallery examples may leave the process in a temporary directory
that is then deleted, breaking subsequent gen-files scripts and rebuilds in
serve mode. Pinning the cwd around each build keeps everything deterministic.
"""

from __future__ import annotations

import os
from pathlib import Path

from mkdocs.plugins import event_priority

_INITIAL_CWD = Path.cwd()


def _chdir() -> None:
    os.chdir(_INITIAL_CWD)


def _reset_gemseo() -> None:
    # Reset GEMSEO global state so any DirectoryManager singleton recreated
    # later does not chdir into a stale (deleted) execution_root_path left
    # behind by a gallery example.
    try:
        from gemseo.utils.global_configuration import _configuration

        _configuration.__init__()
    except Exception:  # noqa: BLE001
        pass


@event_priority(100)
def on_pre_build(config) -> None:  # noqa: ARG001
    _chdir()


@event_priority(100)
def on_files(files, config):  # noqa: ARG001
    _chdir()
    _reset_gemseo()
    return files


@event_priority(100)
def on_post_build(config) -> None:  # noqa: ARG001
    _chdir()
    _reset_gemseo()
