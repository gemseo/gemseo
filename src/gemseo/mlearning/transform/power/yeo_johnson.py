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
r"""A Yeo-Johnson power transformation.

Transform :math:`x` as:

.. math::
   y   = & ((x + 1)^\lambda - 1) / \lambda, \text{ for } x \geq 0, \lambda \neq 0 \\\\
   & \log(x + 1), \text{ for }x \geq 0, \lambda = 0 \\\\
   & -\frac{(1-x)^{2 - \lambda} - 1}{2-\lambda}, \text{for} x < 0, \lambda \neq 2\\\\
   & -log(1-x), \text{ for } x < 0, \lambda = 2

Dependence
----------
This transformation algorithm relies on the ``PowerTransformer`` class
of `scikit-learn <https://scikit-learn.org/
stable/modules/generated/
sklearn.preprocessing.PowerTransformer.html>`_.
"""
from __future__ import annotations

from gemseo.mlearning.transform.power.power import Power


class YeoJohnson(Power):
    """A Yeo-Johnson power transformation."""
