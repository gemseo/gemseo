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

"""
Core classes : disciplines, scenarios, grammars, chains
"""

import os

import matplotlib

from gemseo import LOGGER

# TODO: remove this in favor of export MPLBACKEND=AGG
# To be used for Jenkins which has no display, otherwise graphs export
# wont work
NO_DISPLAY = os.environ.get("NO_DISPLAY")
if NO_DISPLAY == "True":
    LOGGER.warning("Use 'Agg' in matplotlib prevents from displaying windows")
    matplotlib.use("Agg")
