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
Create a first-order Taylor polynomial
======================================

The :class:`.TaylorDiscipline` can be used
to evaluate the first-order polynomial of a discipline
defined at a specific input point.
It can be useful for studies
requiring a lot of evaluations of a discipline,
which is time-consuming but fairly linear around this point.
"""

from __future__ import annotations

from numpy import array

from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.disciplines.taylor import TaylorDiscipline

# %%
# Let us consider a discipline computing :math:`y` from :math:`x=(x_1,x_2)`
# using the function
#
# .. math::
#
#    f(x)=(\sin(x_1)+\cos(x_2),\cos(x_1)+\sin(x_2))
#
discipline = AnalyticDiscipline(
    {"y1": "sin(x1)+cos(x2)", "y2": "cos(x1)+sin(x2)"}, name="f"
)

# %%
# In the following,
# we seek to approximate this model
# using the first-order Taylor polynomial of :math:`f` at an input point :math:`a`:
#
# .. math::
#
#    f_a(x) = \begin{pmatrix}
#              \sin(a_1) + \cos(a_2) + \cos(a_1)(x_1-a_1) - \sin(a_2)(x_2-a_2)\\
#              \cos(a_1) + \sin(a_2) - \sin(a_1)(x_1-a_1) + \cos(a_2)(x_2-a_2)
#             \end{pmatrix}
#
# which could be implemented by hand as follows:
expected_taylor_discipline = AnalyticDiscipline(
    {
        "y1": "sin(a1)+cos(a2)+cos(a1)*(x1-a1)-sin(a2)*(x2-a2)",
        "y2": "cos(a1)+sin(a2)-sin(a1)*(x1-a1)+cos(a2)*(x2-a2)",
    },
    name="expected_f_a",
)

# %%
# Taylor at default input data
# ----------------------------
# For that,
# we can instantiate the :class:`.TaylorDiscipline` from the previous ``discipline``,
# using the ``discipline.input_data`` as the value of :math:`a`:
taylor_discipline = TaylorDiscipline(discipline, name="f_a")
# %%
# We can execute it with its default input data :math:`a`:
taylor_discipline.execute()
# %%
# and compare the results with the expected first-order Taylor polynomial:
expected_taylor_discipline.execute()
print(taylor_discipline.get_output_data(), expected_taylor_discipline.get_output_data())

# %%
# We can also execute it with custom input data
# and compare the results with the expected first-order Taylor polynomial:
taylor_discipline.execute({"x1": array([0.2]), "x2": array([-0.8])})
expected_taylor_discipline.execute({"x1": array([0.2]), "x2": array([-0.8])})
taylor_discipline.get_output_data(), expected_taylor_discipline.get_output_data()

# %%
# Taylor at custom input data
# ---------------------------
# We can also use a custom value of :math:`a`,
# *e.g.* :math:`(0.2, -0.8)` (and compare the results):
taylor_discipline = TaylorDiscipline(
    discipline, name="f_a", input_data={"x1": array([0.2]), "x2": array([-0.8])}
)
# %%
# execute ``taylor_discipline`` with its default input data :math:`a`:
taylor_discipline.execute()
# %%
# and compare the results with the expected first-order Taylor polynomial:
expected_taylor_discipline.default_input_data["a1"] = array([0.2])
expected_taylor_discipline.default_input_data["a2"] = array([-0.8])
expected_taylor_discipline.execute({"x1": array([0.2]), "x2": array([-0.8])})
taylor_discipline.get_output_data(), expected_taylor_discipline.get_output_data()
# %%
# We can also execute it with custom input data
taylor_discipline.execute({"x1": array([1.2]), "x2": array([0.7])})
# %%
# and compare the results with the expected first-order Taylor polynomial:
expected_taylor_discipline.execute({"x1": array([1.2]), "x2": array([0.7])})
taylor_discipline.get_output_data(), expected_taylor_discipline.get_output_data()

# %%
# .. note::
#
#    When the discipline is almost linear over the input range of interest
#    and provides the analytical derivatives,
#    a :class:`.TaylorDiscipline` can be a very relevant surrogate model.
#    Indeed,
#    it can be built with only 1 evaluation
#    whereas a simple linear model would need :math:`1+d` evaluations
#    where :math:`d` is the dimension of the input space.
#
