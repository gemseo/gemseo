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
#        :author: Matthias De Lozzo, Gilberto Ruiz Jimenez
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
r"""A Box-Cox power transformation.

Transform a variable :math:`x` as:

.. math::

   y    & = (x^\lambda - 1) / \lambda,  \text{for } \lambda \neq  0 \\\\
        & = \log(x), \text{for } \lambda = 0

Dependence
----------
This transformation algorithm relies on the ``PowerTransformer`` class
of `scikit-learn <https://scikit-learn.org/
stable/modules/generated/
sklearn.preprocessing.PowerTransformer.html>`_.
"""
from __future__ import annotations

from gemseo.mlearning.transform.power.power import Power
from gemseo.utils.python_compatibility import Final


class BoxCox(Power):
    """A Box-Cox power transformation."""

    _TRANSFORMER_NAME: Final[str] = "box-cox"
