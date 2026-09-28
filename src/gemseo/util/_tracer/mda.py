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
"""Tracer for MDAs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.util._tracer.discipline import DisciplineExecutionTracer

if TYPE_CHECKING:
    from gemseo.mda.core.base_solver import BaseMDASolver
    from gemseo.util.typing import MutableStrKeyMapping
    from gemseo.util.typing import StrKeyMapping


class _BaseMDATracer(DisciplineExecutionTracer):
    """Base tracer for MDA solver related data."""

    _observed_object: BaseMDASolver


class MDAExecutionTracer(_BaseMDATracer):
    """Tracer for MDA solver execution.

    Records trace data for the overall MDA execution lifecycle.
    """


class MDAIterationTracer(_BaseMDATracer):
    """Tracer for MDA solver iteration data.

    Records trace data for each iteration of an MDA solver, including
    the current iteration counter in the trace.
    """

    def _get_input_data(self, call_arguments: StrKeyMapping) -> StrKeyMapping:  # noqa: ARG002
        """Return the input data of the observed iteration.

        The observed method `_iterate_once` takes no argument, so the input
        data is read from the solver's current data, as the solver itself does
        to compute the residuals of an iteration.

        Args:
            call_arguments: The normalized arguments of the observed call,
                by parameter name, unused.

        Returns:
            The input data filtered by the solver's input grammar.
        """
        io = self._observed_object.io
        # TODO: is it really the merged data we want?
        return self._filter_data(io.input_grammar, io.get_merged_data())

    def _get_output_data(self, returned_data: StrKeyMapping | None) -> StrKeyMapping:  # noqa: ARG002
        """Return the output data of the observed iteration.

        The observed method `_iterate_once` returns a stopping indicator instead
        of the output data, so the latter is read from the solver's current
        local output data.

        Args:
            returned_data: The data returned by the observed callable, unused.

        Returns:
            The output data filtered by the solver's output grammar.
        """
        io = self._observed_object.io
        return self._filter_data(io.output_grammar, io.output_data)

    def _get_start_trace(self, call_arguments: StrKeyMapping) -> MutableStrKeyMapping:  # noqa: D102
        trace = super()._get_start_trace(call_arguments)
        trace["iteration"] = self._observed_object._current_iter
        return trace
