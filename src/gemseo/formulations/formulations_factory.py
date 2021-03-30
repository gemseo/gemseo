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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
A factory to instantiate formulation from their class names
***********************************************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from future import standard_library

from gemseo.core.factory import Factory
from gemseo.core.formulation import MDOFormulation

standard_library.install_aliases()


from gemseo import LOGGER


class MDOFormulationsFactory(object):
    """MDO Formulations factory to create the formulation from a name
    or a class.
    """

    def __init__(self):
        """
        Initializes the factory: scans the directories to search for
        subclasses of MDOFormulation.
        Searches in "GEMSEO_PATH" and gemseo.formulations
        """
        # Defines the benchmark problems to be imported
        self.factory = Factory(MDOFormulation, ("gemseo.formulations",))

    def create(
        self, formulation_name, disciplines, objective_name, design_space, **options
    ):
        """
        Create a formulation from its name

        :param formulation_name: the formulation name,
            the class name of the formulation in gemseo.formulations
        :param disciplines: list of disciplines
        :param objective_name: the objective function name
        :param design_space: the design space
        :param options: options for creation of the formulation
        """
        return self.factory.create(
            formulation_name,
            disciplines=disciplines,
            design_space=design_space,
            objective_name=objective_name,
            **options
        )

    @property
    def formulations(self):
        """
        Lists the available classes

        :returns : the list of classes names
        """
        return self.factory.classes

    def is_available(self, formulation_name):
        """
        Checks the availability of a formulation

        :param name : formulation_name of the formulation
        :returns: True if the formulation is available
        """
        return self.factory.is_available(formulation_name)
