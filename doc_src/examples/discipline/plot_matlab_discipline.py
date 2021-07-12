# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under a BSD 0-Clause License.
#
# Permission to use, copy, modify, and/or distribute this software
# for any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
# FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""
Create discipline from a MATLAB function
========================================
"""
###############################################################################
# Import
# ------
from numpy import array

from gemseo.api import create_discipline

###############################################################################
# Simple function with scalar inputs and outputs
# ----------------------------------------------
#
# Assume that we have a MATLAB file :code:`simple_scalar_func.m` that contains
# the following function definition:
#
# .. code::
#
#     function [z1, z2] = simple_scalar_func(x, y)
#     z1 = x^2;
#     z2 = 3*cos(y);
#     end
#
# A very simple convenient way that enables to build a discipline from
# the previous function is to use the |g| API:

disc_sca = create_discipline("MatlabDiscipline", matlab_fct="simple_scalar_func.m")

###############################################################################
# Executing the previous MATLAB discipline is also straightforward:

result = disc_sca.execute({"x": array([2]), "y": array([0.0])})
print(result)

###############################################################################
# Handling input and output vectors
# ---------------------------------
#
# If the discipline involves any vector as input and/or output
# it is quite the same
# than the previous example but one have to be careful with sizes' consistency.
#
# Assume for example the following MATLAB function defined in file
# ``simple_vector_func.m``:
#
# .. code::
#
#     function [z1, z2] = simple_vector_func(x, y)
#     z1(1) = x(1)^2;
#     z1(2) = 2*x(2);
#     z2 = 3*cos(y);
#     end
#
# Thus, inputs must match the right size when executing the discipline:

disc_vec = create_discipline("MatlabDiscipline", matlab_fct="simple_vector_func.m")

result = disc_vec.execute({"x": array([2, 3]), "y": array([0.0])})

print(result)

###############################################################################
# Returning Jacobian matrices
# ---------------------------
#
# For gradient-based optimization, it is usually convenient
# to get access to gradients.
# If gradients are computed inside the MATLAB function, |g| discipline
# can take them into
# account: they just need to be returned properly.
#
# .. note::
#
#     Currently, the computation of the gradient must be in the same MATLAB
#     function than
#     the function itself.
#
# More generally, if the basis function takes an input vector :math:`\bf{x}`
# and returns an
# output vector :math:`\bf{y}`, the total derivatives denoted
# :math:`\frac{d\bf{f}}{d\bf{x}}` is called the jacobian matrix as
# explained in
# :ref:`jacobian_assembly`.
#
# If jacobian matrices are returned by the MATLAB function, |g| discipline can take
# them into account prescribing the argument :code:`is_jac_returned_by_func=True`.
#
# Let's take a simple example and assume that the MATLAB file
# ``jac_fun.m`` contains the following function:
#
# .. code::
#
#     function [ysca, yvec, jac_dysca_dxsca, jac_dysca_dxvec,
#               jac_dyvec_dxsca, jac_dyvec_dxvec] = jac_func(xsca, xvec)
#
#     ysca = xsca + 2*xvec(1) + 3*xvec(2);
#
#     yvec(1) = 4*xsca + 5*xvec(1) + 6*xvec(2);
#     yvec(2) = 7*xsca + 8*xvec(1) + 9*xvec(2);
#
#     jac_dysca_dxsca = 4;
#
#     jac_dysca_dxvec = [2, 3];
#
#     jac_dyvec_dxsca = [4; 7];
#
#     jac_dyvec_dxvec = [[5, 6]; [8, 9]];
#
#     end
#
# Building the discipline is still very simple using the API, we just need to add
# the jacobian boolean argument in this case:


disc_jac = create_discipline(
    "MatlabDiscipline", matlab_fct="jac_func.m", is_jac_returned_by_func=True
)

result = disc_jac.execute({"xsca": array([1]), "xvec": array([2, 3])})

print("Function", result)
print("Jacobians", disc_jac.jac)

###############################################################################
# One can see that jacobian outputs are not included in returned values.
# Since argument ``is_jac_returned_by_func`` has been activated, jacobian matrices
# values are stored in :attr:`MDODiscipline.jac` attributes.
# Some more details about specific options can be found in :ref:`discipline_matlab`.
