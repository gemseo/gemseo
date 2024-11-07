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
Create MDAs with specific settings
----------------------------------

In this example, we show how to create MDAs with specific parameters, such as the
maximum number of iterations or the convergence tolerance.

Let us first create two coupled disciplines, namely the :class:`.Sellar1` and
:class:`.Sellar2` disciplines:
"""

from __future__ import annotations

from gemseo import create_discipline
from gemseo import create_mda

sellar_1, sellar_2 = create_discipline(["Sellar1", "Sellar2"])

# %%
# Using key/value pairs
# ^^^^^^^^^^^^^^^^^^^^^
# A first possibility is to use key/value pairs. In the following code, the
# ``tolerance`` and ``max_mda_iter`` settings are passed as key/value pairs. However,
# this method requires to know the right keyword for the settings, otherwise the
# ``create_mda`` function will raise an error. It is nevertheless relevant for users
# already aware of the keywords.


mda_jacobi = create_mda(
    "MDAJacobi",
    disciplines=[sellar_1, sellar_2],
    max_mda_iter=15,
    tolerance=1e-8,
)

# %%
# Using Pydantic settings model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The second way of providing settings for the MDA is to use the corresponding Pydantic
# settings model. In our example, we intend to create an :class:`.MDAJacobi`, so one
# needs to import the corresponding model:

from gemseo.settings.mda import MDAJacobi_Settings  # noqa: E402

mda_jacobi_settings = MDAJacobi_Settings(max_mda_iter=15, tolerance=1e-8)

mda_jacobi = create_mda(
    "MDAJacobi",
    disciplines=[sellar_1, sellar_2],
    settings_model=mda_jacobi_settings,
)

# %%
# Using Pydantic model to provide settings has to be prefered for two main reasons:
# - The settings are validated (type, value, etc) when the model is created.
# - The auto-complementation and documentation of the model shows all the settings available for the MDA of interest and their **default values**.
#
# The name of the Pydantic model associated with an MDA class is available via the class
# attribute :attr:`~.BaseMDA.Settings`. For instance:

print(mda_jacobi.Settings)


# %%
# Updating the settings
# ---------------------
# It is also possible to update the settings after the MDA object has been created. To
# do so, one must access the settings model attached to each mda via
# :attr:`~.BaseMDA.settings`. Then modifications are as simple as:

new_tolerance = 1e-12
mda_jacobi.settings.tolerance = new_tolerance

# %%
# It is worth knowing that the settings are validated when updated, so the following
# update of ``max_mda_iter`` will raise an error, as the maximum number of iterations
# must be a non-negative integer.

from pydantic_core import ValidationError  # noqa: E402

try:
    mda_jacobi.settings.max_mda_iter = -2
except ValidationError as error:
    print(error)


# %%
# Settings for composed MDAs
# --------------------------
# In |g|, there are two MDAs that are said to be composed because they use or create inner MDAs
# internally, namely :class:`.MDAChain` and :class:`.MDASequential`. For such MDAs,
# the settings are nested, since there are settings for these two classes, and settings
# for the inner-MDAs. Nevertheless, there is a **cascading mechanism** that allows to
# update the settings of the inner-MDAs easily. Let us for instance create an
# ``MDAChain``:

mda_chain = create_mda("MDAChain", disciplines=[sellar_1, sellar_2])

# %%
# By default, the ``MDAChain`` creates ``MDAJacobi`` instances internally. To set the tolerance of
# the inner-MDAs, simply do the following:

inner_mda = mda_chain.inner_mdas[0]
print(f"The tolerance of the inner-MDA is {inner_mda.settings.tolerance}.")
mda_chain.settings.tolerance = 1e-4
print(f"The tolerance of the inner-MDA is now {inner_mda.settings.tolerance}.")
