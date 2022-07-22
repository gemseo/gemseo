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
#                         documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Data conversion between discipline data check and _run()
********************************************************
"""
from __future__ import annotations

import logging

from numpy import array
from numpy import complex128

LOGGER = logging.getLogger(__name__)


class DataProcessor:
    """Abstract class for pre and post processing data of MDODisciplines.

    Executes a pre processing of input data after they are checked by
    MDODiscipline.check_data, and before the _run method of the discipline
    is called.
    Similarly, the post processing method is executed after the _run method
    and before the output data is checked by |g|

    This is usefull to cast the data types, since the |g| kernel only handles
    numpy arrays, and interfaces to disciplinary tools, workflow engines or
    environment can require different types
    """

    def pre_process_data(self, data):
        """Executes a pre processing of input data after they are checked by
        MDODiscipline.check_data, and before the _run method of the discipline is
        called.

        Args:
            data: The input data to process.

        Returns:
            The processed input data.
        """
        raise NotImplementedError()

    def post_process_data(self, data):
        """Executes a post processing of discipline output data after the _run method of
        the discipline, before they are checked by  MDODiscipline.check_output_data,

        Args:
            data: The output data to process.

        Returns:
            The processed output data.
        """
        raise NotImplementedError()


class FloatDataProcessor(DataProcessor):
    """A data preprocessor that converts all gemseo scalar input data to floats, and
    converts all discipline output data to numpy arrays."""

    def pre_process_data(self, data):
        """Executes a pre processing of input data after they are checked by
        MDODiscipline.check_data, and before the _run method of the discipline is
        called.

        Args:
            data: The input data to process.

        Returns:
            The processed input data.
        """
        processed_data = data.copy()
        for key, val in data.items():
            if len(val) == 1:
                processed_data[key] = float(val[0])
            else:
                processed_data[key] = [float(val_i) for val_i in val]
        return processed_data

    def post_process_data(self, data):
        """Executes a post processing of discipline output data after the _run method of
        the discipline, before they are checked by  MDODiscipline.check_output_data,

        Args:
            data: The output data to process.

        Returns:
            The processed output data.
        """
        processed_data = data.copy()
        for key, val in data.items():
            if not hasattr(val, "__len__"):
                processed_data[key] = array([val])
            else:
                processed_data[key] = array(val)
        return processed_data


class ComplexDataProcessor(DataProcessor):
    """A data preprocessor that converts all gemseo complex arrays input data to floats
    arrays, and converts all discipline output data to numpy complex arrays."""

    def pre_process_data(self, data):
        """Executes a pre processing of input data after they are checked by
        MDODiscipline.check_data, and before the _run method of the discipline is
        called.

        Args:
            data: The input data to process.

        Returns:
            The processed input data.
        """
        processed_data = data.copy()
        for key, val in data.items():
            processed_data[key] = array(val.real)
        return processed_data

    def post_process_data(self, data):
        """Executes a post processing of discipline output data after the _run method of
        the discipline, before they are checked by  MDODiscipline.check_output_data,

        Args:
            data: The output data to process.

        Returns:
            The processed output data.
        """
        processed_data = data.copy()
        for key, val in data.items():
            processed_data[key] = array(val, dtype=complex128)
        return processed_data


class NameMapping(DataProcessor):
    """A data preprocessor that maps process level data names to local discipline data
    names."""

    def __init__(self, mapping):
        """
        Args:
            mapping: A mapping structure of the form ``{global_name: local_name}``
                where ``global_name`` must be consistent
                with the grammar of the discipline.
                The local name is the data provided
                to the :meth:`.MDODiscipline._run` method.
        """
        super().__init__()
        self.mapping = mapping
        self.reverse_mapping = {
            local_key: global_key for global_key, local_key in mapping.items()
        }

    def pre_process_data(self, data):
        """Executes a pre processing of input data after they are checked by
        MDODiscipline.check_data, and before the _run method of the discipline is
        called.

        Args:
            data: The input data to process.

        Returns:
            The processed input data.
        """
        mapping = self.mapping
        return {mapping[global_key]: value for global_key, value in data.items()}

    def post_process_data(self, data):
        """Executes a post processing of discipline output data after the _run method of
        the discipline, before they are checked by  MDODiscipline.check_output_data,

        Args:
            data: The output data to process.

        Returns:
            The processed output data.
        """
        reverse_mapping = self.reverse_mapping
        return {reverse_mapping[local_key]: value for local_key, value in data.items()}
