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
#        :author: Damien Guenot
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Abstract factory to create drivers
**********************************
"""
from __future__ import division, unicode_literals

import logging

from gemseo.core.factory import Factory

LOGGER = logging.getLogger(__name__)


class DriverFactory(object):
    """Base class for definition of optimization and/or DOE factory.

    Automates the creation of library interfaces given a name of the algorithm.
    """

    def __init__(self, driver_lib_class, driver_package):
        """Initializes the factory: scans the directories to search for subclasses of
        DriverLib.

        Searches subclasses of driver_lib_class in "GEMSEO_PATH" and driver_package
        """
        self.factory = Factory(driver_lib_class, (driver_package,))
        self.__algo_name_to_lib_name = {}
        self.__update_libdict()

    def __update_libdict(self):
        """Updates the self.__algo_name_to_lib_name dict with available libraries
        list."""
        for lib_name in self.libraries:
            lib = self.create(lib_name)
            for algo_name in lib.algorithms:
                self.__algo_name_to_lib_name[algo_name] = lib_name

    def is_available(self, name):
        """Checks the availability of a library name or algorithm name.

        :param name: the name of the library name or algorithm name
        :returns: True if the library is installed
        """
        is_lib = self.factory.is_available(name)
        return name in self.__algo_name_to_lib_name or is_lib

    @property
    def algorithms(self):
        """Lists the available algorithms names in the present configuration.

        :returns: the list of algorithms as a string list
        """
        return list(self.__algo_name_to_lib_name.keys())

    @property
    def algo_names_to_libraries(self):
        return self.__algo_name_to_lib_name

    @property
    def libraries(self):
        """Lists the available library names in the present configuration.

        :returns: the list of libraries as a string list
        """
        return self.factory.classes

    def create(self, name):
        """Factory method to create a DriverLib subclass from algo identifier or a
        library identifier.

        :param name: library or algorithm name
        :type name: string
        :returns: library according to context (optimization or DOE for instance)
        """
        lib_name = self.__algo_name_to_lib_name.get(name)
        algo_name = None
        if lib_name is not None:
            algo_name = name
        elif self.factory.is_available(name):
            lib_name = name
        else:
            algorithms = sorted(self.algorithms)
            raise ImportError(
                "No algorithm or library of algorithms named '{}' is available; "
                "available algorithms are {}.".format(name, ", ".join(algorithms))
            )
        lib_created = self.factory.create(lib_name)
        # Set the algo name if it was passed by the user as "name" arg
        lib_created.algo_name = algo_name
        return lib_created

    def execute(self, problem, algo_name, **options):
        """Finds the appropriate library and executes the driver on the problem.

        :param problem: the problem on which to run the execution
        :param algo_name: the algorithm name
        :param options: the options dict for the DOE,
            see associated JSON file
        """
        lib = self.create(algo_name)
        return lib.execute(problem, algo_name=algo_name, **options)
