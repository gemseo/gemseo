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
A formulation for uncoupled or weakly coupled problems
******************************************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from builtins import super

from future import standard_library

from gemseo.core.chain import MDOChain
from gemseo.core.execution_sequence import ExecutionSequenceFactory
from gemseo.core.formulation import MDOFormulation
from gemseo.utils.data_conversion import DataConversion

standard_library.install_aliases()


class DisciplinaryOpt(MDOFormulation):
    """
    The disciplinary optimization formulation draws the architecture
    of a mono disciplinary
    optimization process from an ordered list of disciplines,
    an objective function and a design space. The objective function
    is minimized by default.
    """

    def __init__(
        self, disciplines, objective_name, design_space, maximize_objective=False
    ):
        """
        Constructor, initializes the objective functions and constraints

        :param disciplines: the disciplines list.
        :type disciplines: list(MDODiscipline)
        :param objective_name: the objective function data name.
        :type objective_name: str
        :param design_space: the design space.
        :type design_space: DesignSpace
        :param maximize_objective: if True, the objective function
            is maximized, by default, a minimization is performed.
        :type maximize_objective: bool
        """
        self.chain = None
        if len(disciplines) > 1:
            self.chain = MDOChain(disciplines)
        super(DisciplinaryOpt, self).__init__(
            disciplines,
            objective_name,
            design_space,
            maximize_objective=maximize_objective,
        )
        self._filter_design_space()
        self._set_defaultinputs_from_ds()
        # Build the objective from its objective name
        self._build_objective_from_disc(objective_name)

    def get_expected_workflow(self):
        """
        Returns the expected execution sequence,
        used for xdsm representation
        """
        if self.chain is None:
            return ExecutionSequenceFactory.serial(self.disciplines[0])
        return self.chain.get_expected_workflow()

    def get_expected_dataflow(self):
        """
        Returns the expected data exchange sequence,
        used for xdsm representation
        """
        if self.chain is None:
            return []
        return self.chain.get_expected_dataflow()

    def get_top_level_disc(self):
        """Returns the disciplines which inputs are required to run the
        associated scenario
        By default, returns all disciplines
        To be overloaded by subclasses

        :returns: the list of top level disciplines
        """
        if self.chain is not None:
            return [self.chain]
        return self.disciplines

    def _filter_design_space(self):
        """
        Filters the design space to keep only available variables
        """
        all_inpts = DataConversion.get_all_inputs(self.get_top_level_disc())
        kept = set(self.design_space.variables_names) & set(all_inpts)
        self.design_space.filter(kept)
