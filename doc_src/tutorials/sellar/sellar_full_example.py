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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A from scratch example on the Sellar problem."""
from __future__ import annotations

from math import exp

from gemseo.algos.design_space import DesignSpace
from gemseo.api import configure_logger
from gemseo.core.discipline import MDODiscipline
from gemseo.core.mdo_scenario import MDOScenario
from numpy import array
from numpy import ones

configure_logger()


class SellarSystem(MDODiscipline):
    def __init__(self):
        super().__init__()
        # Initialize the grammars to define inputs and outputs
        self.input_grammar.update(["x", "z", "y_0", "y_1"])
        self.output_grammar.update(["obj", "c_1", "c_2"])
        # Default inputs define what data to use when the inputs are not
        # provided to the execute method

    #         self.default_inputs = {"x": ones(1), "z": array([4., 3.]),
    #                                "y_0": ones(1), "y_1": ones(1)}

    def _run(self):
        # The run method defines what happens at execution
        # ie how outputs are computed from inputs
        x, z, y_0, y_1 = self.get_inputs_by_name(["x", "z", "y_0", "y_1"])
        # The ouputs are stored here
        self.local_data["obj"] = array([x[0] ** 2 + z[1] + y_0[0] ** 2 + exp(-y_1[0])])
        self.local_data["c_1"] = array([3.16 - y_0[0] ** 2])
        self.local_data["c_2"] = array([y_1[0] - 24.0])


class Sellar1(MDODiscipline):
    def __init__(self, residual_form=False):
        super().__init__()
        self.input_grammar.update(["x", "z", "y_1"])
        self.output_grammar.update(["y_0"])

    #         self.default_inputs = {"x": ones(1), "z": array([4., 3.]),
    #                                "y_0": ones(1), "y_1": ones(1)}

    def _run(self):
        x, z, y_1 = self.get_inputs_by_name(["x", "z", "y_1"])
        self.local_data["y_0"] = array(
            [(z[0] ** 2 + z[1] + x[0] - 0.2 * y_1[0]) ** 0.5]
        )


class Sellar2(MDODiscipline):
    def __init__(self, residual_form=False):
        super().__init__()
        self.input_grammar.update(["z", "y_0"])
        self.output_grammar.update(["y_1"])

    #         self.default_inputs = {"x": ones(1), "z": array([4., 3.]),
    #                                "y_0": ones(1), "y_1": ones(1)}

    def _run(self):
        z, y_0 = self.get_inputs_by_name(["z", "y_0"])
        self.local_data["y_1"] = array([abs(y_0[0]) + z[0] + z[1]])


def run_process():
    # Instantiate disciplines
    disciplines = [Sellar1(), Sellar2(), SellarSystem()]

    # Creates the design space
    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0.0, u_b=10.0, value=ones(1))
    design_space.add_variable(
        "z", 2, l_b=(-10, 0.0), u_b=(10.0, 10.0), value=array([4.0, 3.0])
    )
    design_space.add_variable("y_0", l_b=-100.0, u_b=100.0, value=ones(1))
    design_space.add_variable("y_1", l_b=-100.0, u_b=100.0, value=ones(1))

    # Build scenario which links the disciplines with the formulation and
    # The optimization algorithm
    scenario = MDOScenario(
        disciplines,
        formulation="MDF",
        objective_name="obj",
        design_space=design_space,
    )

    # Set the design constraints
    scenario.add_constraint("c_1", "ineq")
    scenario.add_constraint("c_2", "ineq")

    # USe finite differences since the disciplines do not provide derivatives
    scenario.set_differentiation_method("finite_differences")

    # Execute scenario with the inputs of the MDO scenario as a dict
    scenario.execute(input_data={"max_iter": 15, "algo": "SLSQP"})

    # Generate a plot of the history in a file
    scenario.post_process("OptHistoryView", save=True, show=True)

    scenario.save_optimization_history(file_path="sellar_history.h5")


if __name__ == "__main__":
    run_process()
