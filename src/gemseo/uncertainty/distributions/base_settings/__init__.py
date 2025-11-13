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
"""Base settings for defining probability distributions.

These settings are abstract Pydantic models
whose `_TARGET_CLASS_NAME` class argument must be defined.
For example,
the Pydantic model
[SPNormalDistribution_Settings][gemseo.uncertainty.distributions.scipy.normal_settings.SPNormalDistribution_Settings]
deriving from
[BaseNormalDistribution_Settings][gemseo.uncertainty.distributions.base_settings.normal_settings.BaseNormalDistribution_Settings]
can be used for defining a normal distribution based on SciPy
(see
[SPNormalDistribution][gemseo.uncertainty.distributions.scipy.normal.SPNormalDistribution]
).
"""
