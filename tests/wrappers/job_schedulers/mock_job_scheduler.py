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
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A mock for the job scheduler executable."""

from __future__ import annotations

from subprocess import run

cmd = (
    r"gemseo-deserialize-run $workdir_path $discipline_path $inputs_path $outputs_path"
)

result = run(cmd, capture_output=True, shell=True)
if result.returncode != 0:
    raise RuntimeError(
        "Failed to execute cmd {}, Received stderr : {}\n, stdout : {}.".format(
            cmd, result.stdout, result.stderr
        )
    )
