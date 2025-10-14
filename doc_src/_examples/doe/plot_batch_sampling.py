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
"""
Batch sampling
==============

GEMSEO v6.2.0 added the option ``vectorize`` (default: ``False``) to the DOE algorithms
to enable batch sampling of outputs of interest.
In other words,
GEMSEO can evaluate a multidisciplinary system at several points at the same time,
without the need for multiple processes
(see the DOE option ``n_processes`` for more information).
This can be particularly useful when evaluating such a system
in parallel is more expensive than evaluating it serially because
the disciplines are so inexpensive.
In this case, the batch sampling can be sequential.

This example illustrates this features,
first for an evaluation problem, then for a scenario.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array
from numpy import newaxis
from pandas.testing import assert_frame_equal

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.scipy.scipy_doe import SciPyDOE
from gemseo.algos.doe.scipy.settings.mc import MC_Settings
from gemseo.algos.evaluation_problem import EvaluationProblem
from gemseo.core.discipline.discipline import Discipline
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.formulations.disciplinary_opt_settings import DisciplinaryOpt_Settings
from gemseo.formulations.mdf_settings import MDF_Settings
from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
from gemseo.problems.mdo.sellar.sellar_design_space import SellarDesignSpace
from gemseo.problems.mdo.sellar.sellar_system import SellarSystem
from gemseo.scenarios.mdo_scenario import MDOScenario
from gemseo.utils.timer import Timer

if TYPE_CHECKING:
    from gemseo.datasets.optimization_dataset import OptimizationDataset
    from gemseo.typing import RealArray
    from gemseo.typing import StrKeyMapping

# %%
# For an evaluation problem
# -------------------------
#
# First,
# we illustrate this feature with an :class:`.EvaluationProblem`
# consisting of calculating the area of several rectangles.
# To do this, we define the design space
# including the bounds of the length and width of such a rectangle:
design_space = DesignSpace()
design_space.add_variable("length", lower_bound=0.0, upper_bound=10.0)
design_space.add_variable("width", lower_bound=0.0, upper_bound=10.0)


# %%
# Then,
# we define a function to compute the area of a rectangle from an array.
# The latter can contain either the values of the length and width of a rectangle,
# i.e., ``array([length, width])``,
# or a collection of ``n`` pairs of length and width corresponding to ``n`` rectangles,
# i.e., ``array([[length_1, width_1], ..., [length_n, width_n]])``.
# Note that the function includes dimension checks to support both cases.


class AreaFunction:
    """Compute the area of a rectangle."""

    def __init__(self) -> None:
        self.last_input_data = array([])

    def __call__(self, input_data: RealArray) -> RealArray:
        """Compute the area of the rectangle.

        Args:
            input_data: The input data.
                This 1D array is
                shaped as ``(input_dimension,)`` or ``(n_samples, input_dimension)``.

        Returns:
            The output data.
            This 1D array is
            shaped as ``(output_dimension,)`` or ``(n_samples, output_dimension)``.
        """
        is_1d = input_data.ndim == 1
        if is_1d:
            input_data = input_data[newaxis, :]

        self.last_input_data = input_data
        length = input_data[:, 0]
        width = input_data[:, 1]
        area = length * width
        return area[0] if is_1d else area


function = AreaFunction()

# %%
# We can verify that the function works correctly:
(
    function(array([3.0, 6.0])),
    function(array([2.0, 4.0])),
    function(array([[3.0, 6.0], [2.0, 4.0]])),
)

# %%
# This function can be attached
# to an :class:`.EvaluationProblem` using an :class:`.MDOFunction`
problem = EvaluationProblem(design_space)
problem.add_observable(MDOFunction(function, "area"))

# %%
# Then,
# this problem can be sampled using a DOE algorithm, e.g., Monte Carlo.
# The default mode is sequential sampling; we use it to generate reference samples:
SciPyDOE("MC").execute(problem, settings_model=MC_Settings(n_samples=1000))
samples_ref = problem.to_dataset()

# %%
# and reset the :class:`.EvaluationProblem` to clear the database before refilling it:
problem.reset()

# %%
# Finally,
# we can generate the same samples in a batch mode
SciPyDOE("MC").execute(
    problem, settings_model=MC_Settings(n_samples=1000, vectorize=True)
)
samples = problem.to_dataset()

# %%
# check that these samples are equal to the reference ones:
assert_frame_equal(samples, samples_ref)

# %%
# and that the function did receive a 2D array of shape ``(1000, 2)``:
assert function.last_input_data.shape == (1000, 2)


# %%
# For a scenario
# --------------
#
# In this second part of the example,
# we will apply batch sampling to the case of a scenario.
#
# Vectorizing the user disciplines is more complicated than vectorizing functions
# because the user disciplines will be handled by multidisciplinary processes,
# including process disciplines,
# which only accept 1D input and output arrays and 2D Jacobian arrays.
# A user discipline must therefore always receive dictionaries of 1D arrays
# and return dictionaries of 1D arrays.
#
class AreaDiscipline(Discipline):
    """Compute the area of a rectangle."""

    def __init__(self) -> None:
        super().__init__()
        self.io.input_grammar.update_from_names(("length", "width"))
        self.io.output_grammar.update_from_names(("area",))

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        length = input_data["length"]
        width = input_data["width"]
        return {"area": length * width}


area_discipline = AreaDiscipline()


# %%
# In the case of this discipline,
# the expected forms of the input and output data are
# ``{"length": array([length_1, ..., length_n]), "width": array([width_1, ..., width_n])}``
# and ``{"area": array([area_1, ..., area_n])}`` respectively.
#
# Note that in the case of a ``d``-length vectorial variable ``"x"``,
# the expected form would be
# ``{"x": array([x_1_1, ..., x_d_1, ..., x_1_n, ..., x_d_n])}``.
#
# DisciplinaryOpt formulation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We use this discipline to illustrate the batch sampling feature
# with the :class:`.DisciplinaryOpt` formulation.
# Remember that this MDO formulation executes the disciplines
# in the order provided by the user.
# For a better illustration,
# we add a discipline to post-process the output data of ``AreaDiscipline``:
class AreaIncreaser(Discipline):
    """Increase the area by a factor of 2."""

    def __init__(self) -> None:
        super().__init__()
        self.io.input_grammar.update_from_names(("area",))
        self.io.output_grammar.update_from_names(("final_area",))

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        return {"final_area": 2.0 * input_data["area"]}


disciplines = [area_discipline, AreaIncreaser()]

# %%
# Now,
# we can create and execute the scenario using batch sampling:
scenario = MDOScenario(
    disciplines,
    "final_area",
    design_space,
    formulation_settings_model=DisciplinaryOpt_Settings(),
)
scenario.execute(MC_Settings(n_samples=1000, vectorize=True))
samples = scenario.to_dataset(opt_naming=False)

# %%
# We can see that the samples are equal to the reference ones to within a factor of 2:
samples_ref.rename_variable("area", "final_area")
samples_ref.transform_data(lambda area: 2.0 * area, variable_names="final_area")
assert_frame_equal(samples, samples_ref)

# %%
# and that the ``AreaDiscipline`` received and returned 1D arrays of size 1000:
assert area_discipline.io.data["length"].shape == (1000,)
assert area_discipline.io.data["width"].shape == (1000,)
assert area_discipline.io.data["area"].shape == (1000,)


# %%
# MDF formulation
# ~~~~~~~~~~~~~~~
# To conclude this example,
# we will illustrate the applicability of batch sampling to the MDF formulation,
# including the computation of Jacobian data.
#
# Let us consider the Sellar MDO problem whose disciplines have been vectorized
# (look at the source code of :class:`.Sellar1`, :class:`.Sellar2`
# and :class:`.SellarSystem`,
# including the part related to the Jacobian computation with sparse arrays.).
#
# We create a function defining and solving this MDO problem
# using either sequential or batch sampling:
def solve_sellar(
    use_batch_sampling: bool, eval_jac: bool
) -> tuple[OptimizationDataset, float]:
    """Solve the Sellar sampling problem.

    Args:
        use_batch_sampling: Whether to use batch sampling.
        eval_jac: Whether to sample the Jacobian functions.

    Returns:
        The samples, the execution time.
    """
    disciplines = [Sellar1(), Sellar2(), SellarSystem()]
    design_space = SellarDesignSpace(dtype="float")
    scenario = MDOScenario(
        disciplines, "obj", design_space, formulation_settings_model=MDF_Settings()
    )
    scenario.add_constraint("c_1")
    scenario.add_constraint("c_2")
    with Timer() as timer:
        scenario.execute(
            MC_Settings(n_samples=100, vectorize=use_batch_sampling, eval_jac=eval_jac)
        )

    return scenario.to_dataset(), timer.elapsed_time


# %%
# solve it with both sampling modes:
samples_ref, time_ref = solve_sellar(False, False)
samples, time = solve_sellar(True, False)

# %%
# check that the samples are equal:
assert_frame_equal(samples, samples_ref, rtol=1e-3)

# %%
# and see a significant execution time reduction:
f"{round((time_ref - time) / time_ref * 100)} %"

# %%
# The same type of results are obtained by evaluating the gradients.
samples_ref, time_ref = solve_sellar(False, True)
samples, time = solve_sellar(True, True)
assert_frame_equal(samples, samples_ref, rtol=1e-3)

# %%
# However, the gain is slightly lower (but still represents a significant gain!),
# which may be counterintuitive and will be the subject of future investigations:
f"{round((time_ref - time) / time_ref * 100)} %"
