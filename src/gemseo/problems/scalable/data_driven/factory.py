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
#    INITIAL AUTHORS - initial API and implementation and/or
#                  initial documentation
#        :author:  Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Scalable model factory.

This module contains the :class:`.ScalableModelFactory` which is a factory
to create a :class:`.ScalableModel` from its class name by means of the
:meth:`.ScalableModelFactory.create` method. It is also possible to get a list
of available scalable models
(see :attr:`.ScalableModelFactory.scalable_models` method)
and to check is a type of scalable model is available
(see :meth:`.ScalableModelFactory.is_available` method)
"""

from __future__ import annotations

from gemseo.core.base_factory import BaseFactory
from gemseo.problems.scalable.data_driven.model import ScalableModel


class ScalableModelFactory(BaseFactory):
    """This factory instantiates a class:`.ScalableModel` from its class name.

    The class can be internal to |g| or located in an external module whose path is
    provided to the constructor.
    """

    _CLASS = ScalableModel
    _MODULE_NAMES = ("gemseo.problems.scalable",)

    def create(self, model_name: str, data, sizes=None, **parameters):
        """Create a scalable model.

        :param str model_name: name of the scalable model (its class name)
        :param Dataset data: learning dataset.
        :param dict sizes: sizes of input and output variables.
            If ``None``, use the original sizes.
            Default: None.
        :param parameters: model parameters
        :return: model_name scalable model
        """
        return super().create(model_name, data=data, sizes=sizes, **parameters)

    @property
    def scalable_models(self) -> list[str]:
        """Lists the available classes for scalable models.

        :returns: the list of classes names.
        :rtype: list(str)
        """
        return self.class_names
