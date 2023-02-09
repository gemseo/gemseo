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
# -*-mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8 -*-
# Copyright (c) 2018 IRT-AESE.
# All rights reserved.
#
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Arthur Piat
#        :author: Nicolas Roussouly: GEMSEO integration
"""Definition of the Matlab license manager.

Overview
--------

This module contains the :class:`.LicenseManager`
which enables to check the presence of any toolbox licenses
of the Matlab installation.
"""
from __future__ import annotations

import logging
from pathlib import Path

from gemseo.wrappers.matlab.engine import MatlabEngine

LOGGER = logging.getLogger(__name__)


class LicenseManager:
    """Manage Matlab License.

    The licenseManager was created to enable de-synchronised
    launch of optimization using matlab_discipline. The goal
    is to wait until all licenses that are needed are available
    in Matlab workspace.
    Parallel computing launch can be used with this tool.

    Examples:
        >>> # Build a new matlab engine
        >>> eng = get_matlab_engine()
        >>> # add a toolbox to the engine
        >>> eng.add_toolbox("signal_toolbox")
        >>> # build a license manager from the previous engine
        >>> lm = LicenseManager(eng)
        >>> # check licenses of the engine until all are available
        >>> lm.check_licenses()
    """

    engine: MatlabEngine
    """The MatlabEngine instance."""

    SIGNAL_TOOL = "signal_toolbox"
    DISTRIB_COMP_TOOL = "distrib_computing_toolbox"
    CURVE_FIT_TOOL = "Curve_Fitting_Toolbox"

    def __init__(self, engine: MatlabEngine) -> None:
        """
        Args:
            engine: The MatlabEngine instance.
        """  # noqa: D205, D212, D415
        self.__engine = engine
        if self.__engine.is_closed:
            # If engine is not (re)started here, add_path
            # will raise an error. A closed engine could happen here
            # since engine is a singleton and could be closed in any
            # other location
            self.__engine.start_engine()

        self.__engine.add_path(str(Path(__file__).parent / "matlab_files"))
        self.__licenses = self.__engine.get_toolboxes()

    @property
    def licenses(self):
        """The licences."""
        return self.__licenses

    @licenses.setter
    def licenses(self, licenses):
        self.__licenses = licenses

    def check_licenses(
        self,
        licenses: list[str] | None = None,
        pause_frac: float = 60,
        pause_const: float = 20,
    ) -> None:
        """Check that the Matlab licenses exist.

        The method fetches all the needed licenses thanks to the
        matlab function licenseControl and the class Logger. Note
        that the MATLAB function will be looping until the given
        toolboxes are available.

        Args:
            licenses: The list of matlab toolboxes.
                If ``None``, use the already existing engine licenses.
            pause_frac: The time used between each try to get licenses.
            pause_const: The time added in order to estimate the waiting time.
                The waiting time is estimated at each try with the following formula:
                ``Wt = pause_const + random([0,1])*pause_frac``.
        """
        if licenses is not None:
            self.licenses = licenses

        if self.licenses is None or not self.licenses:
            LOGGER.info("No MATLAB license check will be performed.")
            return

        self.__engine.execute_function(
            "licenseControl", self.licenses, pause_frac, pause_const, nargout=0
        )

    def start_parallel_computing(
        self,
        n_parallel_workers: int = 4,
        cluster_name: str = "local",
    ) -> bool:
        """Start parallel computing in MatlabEngine.

        Args:
            n_parallel_workers: The number of "workers" to the parallel pool.
                Maximum number allowed is 12
            cluster_name: The matlab parallel pool cluster name. If None,
                the given name is the same as the engine.
        """
        self.check_licenses(self.DISTRIB_COMP_TOOL)
        self.__engine.start_parallel_computing(n_parallel_workers, cluster_name)

    def end_parallel_computing(self) -> bool:
        """Stop parallel computing in matlab engine."""
        self.__engine.end_parallel_computing()
