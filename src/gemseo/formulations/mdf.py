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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
The Multi-disciplinary Design Feasible formulation
**************************************************
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library

from gemseo.core.formulation import MDOFormulation
from gemseo.mda.mda_factory import MDAFactory

standard_library.install_aliases()


class MDF(MDOFormulation):
    """
    The Multidisciplinary Design Feasible formulation draws an
    optimization architecture where the coupling of strongly
    coupled disciplines is made consistent by means of a
    Multidisciplinary Design Analysis (MDA), the optimization
    problem w.r.t. local and global design variables is made
    at the top level. Multidisciplinary analysis is made at
    a each optimization iteration.
    """

    def __init__(
        self,
        disciplines,
        objective_name,
        design_space,
        maximize_objective=False,
        main_mda_class="MDAChain",
        sub_mda_class="MDAJacobi",
        **mda_options
    ):
        """
        Constructor, initializes the objective functions and constraints

        :param main_mda_class: classname of the main MDA, typically the
            MDAChain,  but one can force to use MDAGaussSeidel for instance
        :type main_mda_class: str
        :param disciplines: the disciplines list.
        :type disciplines: list(MDODiscipline)
        :param objective_name: the objective function data name.
        :type objective_name: str
        :param design_space: the design space.
        :type design_space: DesignSpace
        :param maximize_objective: if True, the objective function
            is maximized, by default, a minimization is performed.
        :type maximize_objective: bool
        :param sub_mda_class: the type of MDA  to be used,
            shall be the class name. (default MDAJacobi)
        :type sub_mda_class: str
        :param mda_options: options passed to the MDA at construction
        """
        super(MDF, self).__init__(
            disciplines,
            objective_name,
            design_space,
            maximize_objective=maximize_objective,
        )
        self.mda = None
        self._main_mda_class = main_mda_class
        self._mda_factory = MDAFactory()
        self._instantiate_mda(main_mda_class, sub_mda_class, **mda_options)
        self._update_design_space()
        self._build_objective()

    def get_top_level_disc(self):
        return [self.mda]

    def _instantiate_mda(
        self, main_mda_class="MDAChain", sub_mda_class="MDAJacobi", **mda_options
    ):
        """Create MDA discipline"""
        if main_mda_class == "MDAChain":
            mda_options["sub_mda_class"] = sub_mda_class
        self.mda = self._mda_factory.create(
            main_mda_class, self.disciplines, **mda_options
        )

    @classmethod
    def get_sub_options_grammar(cls, **options):
        """
        When some options of the formulation depend on higher level
        options, a sub option schema may be specified here, mainly for
        use in the API

        :param options: options dict required to deduce the sub options grammar
        :returns: None, or the sub options grammar
        """
        main_mda = options.get("main_mda_class")
        if main_mda is None:
            raise ValueError(
                "main_mda_class option required \n"
                + "to deduce the sub options of MDF !"
            )
        factory = MDAFactory().factory
        return factory.get_options_grammar(main_mda)

    @classmethod
    def get_default_sub_options_values(cls, **options):
        """
        When some options of the formulation depend on higher level
        options, a sub option defaults may be specified here, mainly for
        use in the API

        :param options: options dict required to deduce the sub options grammar
        :returns: None, or the sub options defaults
        """
        main_mda = options.get("main_mda_class")
        if main_mda is None:
            raise ValueError(
                "main_mda_class option required \n"
                + "to deduce the sub options of MDF !"
            )
        factory = MDAFactory().factory
        return factory.get_default_options_values(main_mda)

    def _build_objective(self):
        """Builds the objective function on the MDA"""
        # Build the objective from the mda and the objective name
        self._build_objective_from_disc(self._objective_name, discipline=self.mda)

    def get_expected_workflow(self):
        return self.mda.get_expected_workflow()

    def get_expected_dataflow(self):
        return self.mda.get_expected_dataflow()

    def _update_design_space(self):
        """Update the design space by removing the coupling variables"""
        self._set_defaultinputs_from_ds()
        # No couplings in design space (managed by MDA)
        self._remove_couplings_from_ds()
        # Cleanup
        self._remove_unused_variables()

    def _remove_couplings_from_ds(self):
        """Removes the coupling variables from the design space"""
        design_space = self.opt_problem.design_space
        for coupling in self.mda.strong_couplings:
            if coupling in design_space.variables_names:
                design_space.remove_variable(coupling)
