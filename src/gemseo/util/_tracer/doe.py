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
"""Tracer for DOEs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.util._tracer.base import BaseTracer

if TYPE_CHECKING:
    from gemseo.doe.core.base_doe_library import BaseDOELibrary
    from gemseo.util.typing import MutableStrKeyMapping
    from gemseo.util.typing import StrKeyMapping


class DOETracer(BaseTracer):
    """Tracer for recording DOE algorithm evaluation data.

    Records trace information for each sample point evaluation in a DOE.
    """

    _observed_object: BaseDOELibrary

    def _get_start_trace(self, call_arguments: StrKeyMapping) -> MutableStrKeyMapping:  # noqa: D102
        trace = super()._get_start_trace(call_arguments)
        # The arguments are read with a default rather than indexed, as
        # `DOEDMProcessor.start` does for the sample index: a parameter may
        # simply be absent from a variadic signature, or a subclass may
        # rename it, and the trace shall then lose it instead of breaking
        # the observed call.
        # A negative sample index means that all the samples are evaluated at
        # once, see BaseDOELibrary._evaluate_functions.
        trace["sample_index"] = call_arguments.get("sample_index", -1)
        if "input_value" in call_arguments:
            trace["input_value"] = call_arguments["input_value"]
        return trace
