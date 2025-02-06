# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under a BSD 0-Clause License.
#
# Permission to use, copy, modify, and/or distribute this software
# for any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
# FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Fabian Castañeda
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Compute the Jacobian of a discipline with analytical and approximated elements
==============================================================================

In this example,
we will compute the Jacobians of some outputs of an :class:`.Discipline`
with respect to some inputs, based on some analytical derivatives and approximative
methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array

from gemseo.core.discipline import Discipline

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo import StrKeyMapping

# %%
# For many different reasons, one might be in a situation where not all the derivatives
# of a given discipline are at hand and approximating all of them might not be
# convenient for a reason or another. For situations like these, being able to compute
# the Jacobian of a discipline using both analytical expressions for certain
# inputs-outputs and approximative methods for the rest can be handy.
# First,
# we create a discipline, e.g. a :class:`.Discipline`:


class HybridDiscipline(Discipline):
    def __init__(self) -> None:
        super().__init__()
        self.io.input_grammar.update_from_names(["x_1", "x_2", "x_3"])
        self.io.output_grammar.update_from_names(["y_1", "y_2", "y_3"])
        self.io.input_grammar.defaults = {
            "x_1": array([1.0]),
            "x_2": array([2.0]),
            "x_3": array([1.0]),
        }

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        y_1 = input_data["x_1"] * input_data["x_2"]
        y_2 = input_data["x_1"] * input_data["x_2"] * input_data["x_3"]
        y_3 = input_data["x_1"]
        return {"y_1": y_1, "y_2": y_2, "y_3": y_3}

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        self._init_jacobian()
        x1 = array([self.get_input_data(with_namespaces=False)["x_1"]])
        x2 = array([self.get_input_data(with_namespaces=False)["x_2"]])
        x3 = array([self.get_input_data(with_namespaces=False)["x_3"]])
        self.jac = {"y_1": {"x_1": x2}, "y_2": {"x_2": x1 * x3}}


# %%
# As you can see, we define the jacobian of the discipline inside the discipline's method
# :meth:`._compute_jacobian`. However, we are only defining the derivatives that
# we have or care about.

# %%
# In this case we define ``"y_1"`` wrt ``"x_1"`` and ``"y_2"`` wrt to ``"x_2"``.
# This means that we are missing ``"y_1"`` wrt to ``"x_2"``, ``"y_2"`` wrt to ``"x_1"``
# and ``"x_3"`` and finally ``"y_3"`` wrt ``"x_1"``.
# we can call the discipline's method :meth:`.linearize` to fill in the missing
# derivatives. Nonetheless, we need to parametrized it to just compute the missing
# derivatives. For this we assign to the attribute :attr:`.linearization_mode` one of
# the hybrid available modes which are accessible from the attribute
# :attr:`.ApproximationMode`.


discipline = HybridDiscipline()
discipline.linearization_mode = discipline.ApproximationMode.HYBRID_FINITE_DIFFERENCES

# %%
# There are three modes available, ``HYBRID_FINITE_DIFFERENCES``,
# ``HYBRID_CENTERED_DIFFERENCES`` and ``HYBRID_COMPLEX_STEP``. Being the difference
# between each other the approximation type used to approximate the missing derivatives.
# We can also define the inputs to be used to compute the Jacobian, in this case we are
# using the default inputs. Finally, we need to set the ``"compute_all_jacobians"`` flag
# to True. Even if we are not computing them all, this option needs to be active in
# order to access the data for the hybrid linearization.

inputs = discipline.default_input_data
jacobian_data = discipline.linearize(inputs, compute_all_jacobians=True)
jacobian_data
