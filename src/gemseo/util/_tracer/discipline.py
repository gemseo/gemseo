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
"""Tracer for disciplines."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from gemseo.core.cache.hdf5 import HDF5Cache
from gemseo.util._tracer.base import BaseTracer
from gemseo.util.constant import read_only_empty_dict

if TYPE_CHECKING:
    from gemseo.core.discipline.discipline import Discipline
    from gemseo.core.grammar.base import BaseGrammar
    from gemseo.util.typing import MutableStrKeyMapping
    from gemseo.util.typing import StrKeyMapping


class _BaseDisciplineTracer(BaseTracer):
    """Base tracer for discipline related data.

    Shared by the tracers observing discipline executions, discipline
    linearizations and MDA solvers, all of which record data filtered by a
    discipline's input and output grammars.
    """

    _observed_object: Discipline

    def _get_start_trace(self, call_arguments: StrKeyMapping) -> MutableStrKeyMapping:  # noqa: D102
        trace = super()._get_start_trace(call_arguments)
        trace["input_data"] = self._get_input_data(call_arguments)
        return trace

    def _get_input_data(self, call_arguments: StrKeyMapping) -> StrKeyMapping:
        """Return the input data of the observed call.

        The observed methods `execute` and `linearize` both have an
        `input_data` parameter, whose default value is empty; the arguments
        are normalized, so this parameter can be read whatever the way the
        caller passed it, or did not pass it.

        The parameter is read with a default rather than indexed: it may
        simply be absent from `call_arguments`, e.g. a truly positional
        variadic signature has no parameter named `input_data` at all, or a
        subclass may rename the parameter; the trace shall then lose the
        input data instead of breaking the observed call.

        Args:
            call_arguments: The normalized arguments of the observed call,
                by parameter name.

        Returns:
            The input data filtered by the discipline's input grammar.
        """
        return self._filter_data(
            self._observed_object.io.input_grammar,
            call_arguments.get("input_data", read_only_empty_dict),
        )

    def _get_output_data(self, returned_data: StrKeyMapping | None) -> StrKeyMapping:
        """Return the output data of the observed call.

        The observed method `execute` returns the output data, so it can be read
        from the returned data. When the call raised,
        `injector._end_observation_safely` passes no returned data, and the
        discipline's current output data is used instead, reflecting whatever
        was produced before the failure.

        Args:
            returned_data: The data returned by the observed callable,
                or `None` if the call raised.

        Returns:
            The output data filtered by the discipline's output grammar.
        """
        io = self._observed_object.io
        data = io.output_data if returned_data is None else returned_data
        return self._filter_data(io.output_grammar, data)

    @staticmethod
    def _filter_data(grammar: BaseGrammar, data: StrKeyMapping) -> StrKeyMapping:
        """Remove data items that are not in the grammar.

        Args:
            grammar: The grammar.
            data: The data.

        Returns:
            The filtered data, in the order of `data`, not of the grammar.
        """
        return {name: data[name] for name in data if name in grammar}


class DisciplineExecutionTracer(_BaseDisciplineTracer):
    """Tracer for recording discipline execution data.

    Captures input data and output data based on the discipline's grammars,
    and records cache file paths for HDF5 caches.
    """

    def _get_start_trace(self, call_arguments: StrKeyMapping) -> MutableStrKeyMapping:  # noqa: D102
        trace = super()._get_start_trace(call_arguments)
        if isinstance(self._observed_object.cache, HDF5Cache):
            # The cache can be (re)configured after construction, e.g. via
            # Discipline.set_cache, so its path is read here, at call time,
            # instead of being recorded once at registration.
            trace["cache_path"] = self._observed_object.cache.hdf_file.hdf_file_path
        return trace

    def _get_end_trace(
        self, call_arguments: StrKeyMapping, returned_data: Any
    ) -> MutableStrKeyMapping:  # noqa: D102
        trace = super()._get_end_trace(call_arguments, returned_data)
        trace["output_data"] = self._get_output_data(returned_data)
        return trace


class DisciplineLinearizationTracer(_BaseDisciplineTracer):
    """Tracer for recording discipline linearization data.

    Captures input data based on the discipline's input grammar,
    and the resulting Jacobian data.
    """

    def _get_end_trace(
        self, call_arguments: StrKeyMapping, returned_data: Any
    ) -> MutableStrKeyMapping:  # noqa: D102
        trace = super()._get_end_trace(call_arguments, returned_data)
        trace["jacobian"] = returned_data
        return trace
