# -*- coding: utf-8 -*-
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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                        documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Scenario which drivers are Design of Experiments
************************************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from future import standard_library

from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.core.scenario import Scenario

standard_library.install_aliases()


# The detection of formulations requires to import them,
# before calling get_formulation_from_name
from gemseo import LOGGER


class DOEScenario(Scenario):
    """Design of Experiments scenario, based on MDO scenario but with a DOE
    driver.

    The main differences between Scenario and MDOScenario are the allowed
    inputs
    in the MDOScenario.json, which differs from DOEScenario.json, at least
    on the driver names

    MDO Problem description: links the disciplines and the formulation
    to create an optimization problem.
    Use the class by instantiation.
    Create your disciplines beforehand.

    Specify the formulation by giving the class name such as the string
    "MDF"

    The reference_input_data is the typical input data dict that is provided
    to the run method of the disciplines

    Specify the objective function name, which must be an output
    of a discipline of the scenario, with the "objective_name" attribute

    If you want to add additional design constraints,
    use the add_user_defined_constraint method

    To view the results, use the "post_process" method after execution.
    You can view:

    - the design variables history, the objective value, the constraints,
      by using: scenario.post_process("OptHistoryView", show=False, save=True)
    - Quadratic approximations of the functions close to the
      optimum, when using gradient based algorithms, by using:
      scenario.post_process("QuadApprox", method="SR1", show=False,
      save=True, function="my_objective_name",
      file_path="appl_dir")
    - Self Organizing Maps of the design space, by using:
      scenario.post_process("SOM", save=True, file_path="appl_dir")

    To list post processings on your setup,
    use the method :attr:`.Scenario.posts`.
    For more details on their options, go to the **gemseo.post** package.
    """

    # Constants for input variables in json schema
    N_SAMPLES = "n_samples"
    EVAL_JAC = "eval_jac"

    def __init__(
        self,
        disciplines,
        formulation,
        objective_name,
        design_space,
        name=None,
        **formulation_options
    ):
        """Constructor, initializes the DOE scenario
        Objects instantiation and checks are made before run intentionally

        :param disciplines: the disciplines of the scenario
        :param formulation: the formulation name,
            the class name of the formulation in gemseo.formulations
        :param objective_name: the objective function name
        :param design_space: the design space
        :param name: scenario name
        :param formulation_options: options for creation of the formulation
        """
        # This loads the right json grammars from class name
        super(DOEScenario, self).__init__(
            disciplines,
            formulation,
            objective_name,
            design_space,
            name,
            **formulation_options
        )

        self.default_inputs = {self.EVAL_JAC: False, self.ALGO: "lhs"}

    def _init_algo_factory(self):
        """
        Initalizes the algorithms factory
        """
        self._algo_factory = DOEFactory()

    def _run_algorithm(self):
        """Runs the DOE algo"""
        problem = self.formulation.opt_problem
        algo_name = self.local_data[self.ALGO]
        n_samples = self.local_data.get(self.N_SAMPLES)
        options = self.local_data.get(self.ALGO_OPTIONS)
        if options is None:
            options = {}
        if self.N_SAMPLES in options:
            LOGGER.warning(
                "Double definition of algorithm option n_samples, " "keeping value: %s",
                n_samples,
            )
            options.pop(self.N_SAMPLES)

        if self.ALGO_OPTIONS in self.local_data:
            options = self.local_data[self.ALGO_OPTIONS]
        lib = self._algo_factory.create(algo_name)

        self.optimization_result = lib.execute(problem, n_samples=n_samples, **options)
        return self.optimization_result

    def _run(self):
        """Execute the scenario and run the optimization problems"""
        LOGGER.info(" ")
        LOGGER.info("*** Start DOE Scenario execution ***")
        self.log_me()
        self._run_algorithm()
        LOGGER.info("*** DOE Scenario run terminated ***")
