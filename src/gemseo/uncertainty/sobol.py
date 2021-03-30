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
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

"""
Sobol' indices
==============
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from numpy import array
from openturns import (
    JansenSensitivityAlgorithm,
    MartinezSensitivityAlgorithm,
    MauntzKucherenkoSensitivityAlgorithm,
    SaltelliSensitivityAlgorithm,
    Sample,
)

from gemseo.core.discipline import MDODiscipline
from gemseo.core.doe_scenario import DOEScenario
from gemseo.utils.data_conversion import DataConversion

standard_library.install_aliases()

from gemseo import LOGGER


class SobolIndices(object):
    """ Sobol' indices. """

    ALGOS = {
        "Saltelli": SaltelliSensitivityAlgorithm,
        "Jansen": JansenSensitivityAlgorithm,
        "MauntzKucherenko": MauntzKucherenkoSensitivityAlgorithm,
        "Martinez": MartinezSensitivityAlgorithm,
    }

    def __init__(
        self, disciplines, space, n_samples, formulation="MDF", objective_name=None
    ):
        """Constructor

        :param list(MDODiscipline) disciplines: disciplines.
        :param ParameterSpace space: parameter space.
        :param int n_samples: number of samples.
        :param str formulation: MDO formulation. Default: 'MDF'.
        :param str objective_name: objective name. If None, use the first
            output. Default: None.
        """
        if isinstance(disciplines, MDODiscipline):
            disciplines = [disciplines]
        first_output = next(iter(disciplines[0].get_output_data_names()))
        objective_name = objective_name or first_output
        scenario = DOEScenario(disciplines, formulation, objective_name, space)
        scenario.execute({"algo": "OT_SOBOL_INDICES", "n_samples": n_samples})
        opt_problem = scenario.formulation.opt_problem
        self.dataset = opt_problem.export_to_dataset("sobol", opt_naming=False)
        self.n_samples = len(self.dataset)

    def get_indices(self, algo="Saltelli"):
        """Get Sobol' indices.

        :param str algo: method to compute the Sobol' indices,
            either 'Saltelli', 'Jansen', 'MauntzKucherenko' or 'Martinez'.
            Default: 'Saltelli'.
        """
        try:
            algo = self.ALGOS[algo]
        except Exception:
            raise TypeError(
                algo + " is not an available algorithm " "to compute Sobol" " indices."
            )
        array_to_dict = DataConversion.array_to_dict
        inputs_names = self.dataset.get_names("inputs")
        sizes = self.dataset.sizes
        inputs = Sample(self.dataset.get_data_by_group("inputs"))
        outputs = Sample(self.dataset.get_data_by_group("outputs"))
        dim = self.dataset.dimension[self.dataset.INPUT_GROUP]
        n_samples = int(self.n_samples / (dim + 2))
        sobol = algo(inputs, outputs, n_samples)
        first_order = array(sobol.getFirstOrderIndices())
        first_order = array_to_dict(first_order, inputs_names, sizes)
        total_order = array(sobol.getTotalOrderIndices())
        total_order = array_to_dict(total_order, inputs_names, sizes)
        return sobol, first_order, total_order
