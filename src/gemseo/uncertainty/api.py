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
API for uncertainty management
==============================

This API (Application Programming Interface) is dedicated to
uncertainty management. Current functions are:

- :meth:`~gemseo.uncertainty.api.get_available_distributions`,
- :meth:`~gemseo.uncertainty.api.create_distribution`,
- :meth:`~gemseo.uncertainty.api.create_statistics`.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library

standard_library.install_aliases()

# pylint: disable=import-outside-toplevel


def get_available_distributions():
    """Get the available distributions."""
    from gemseo.uncertainty.distributions.factory import DistributionFactory

    factor = DistributionFactory()
    return factor.distributions


def create_distribution(variable, distribution_type, dimension=1, **options):
    """Create a distribution

    :param str variable: variable name.
    :param str distribution_type: distribution type.
    :param int dimension: variable dimension.
    :param options: distribution options.
    """
    from gemseo.uncertainty.distributions.factory import DistributionFactory

    factor = DistributionFactory()
    return factor.create(
        distribution_type, variable=variable, dimension=dimension, **options
    )


def create_statistics(
    dataset,
    variables_names=None,
    tested_distributions=None,
    fitting_criterion="BIC",
    selection_criterion="best",
    level=0.05,
    name=None,
):
    """Constructor

    :param Dataset dataset: dataset
    :param list(str) variables_names: list of variables names
        or list of variables names. If None, the method considers
        all variables from loaded dataset. Default: None.
    :param list(str) tested_distributions: list of candidate distributions
        names for parametric statistics. If None, considers empirical
        statistics. Default: None.
    :param str fitting_criterion: goodness-of-fit criterion
        for parametric statistics. Default: 'BIC'.
    :param float level: risk of committing a Type 1 error,
        that is an incorrect rejection of a true null hypothesis,
        for criteria based on test hypothesis in the case of
        parametric statistics. Default: 0.05.
    :param str selection_criterion: selection criterion
        for parametric statistics. Default: 'best'
    :param str name: name of the object.
        If None, use the concatenation of class and dataset names.
        Default: None.
    """
    from gemseo.uncertainty.statistics.empirical import EmpiricalStatistics as EmpStats
    from gemseo.uncertainty.statistics.parametric import (
        ParametricStatistics as ParamStats,
    )

    if tested_distributions is None:
        statistical_analysis = EmpStats(dataset, variables_names, name)
    else:
        statistical_analysis = ParamStats(
            dataset,
            tested_distributions,
            variables_names,
            fitting_criterion,
            level,
            selection_criterion,
            name,
        )
    return statistical_analysis
