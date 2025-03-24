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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Scalers
=======
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from numpy import linspace

from gemseo.mlearning.transformers.scaler.min_max_scaler import MinMaxScaler
from gemseo.mlearning.transformers.scaler.scaler import Scaler
from gemseo.mlearning.transformers.scaler.standard_scaler import StandardScaler

# %%
# Scaling data may be important, as discussed in another example.
# Different scalers are available
# and this example illustrate them with these simple data:
data = linspace(-2, 2, 100)

# %%
# First,
# a :class:`.Scaler` transforms a value :math:`x` into a new value :math:`\tilde{x}`
# based on the linear function :math:`\tilde{x}=a+bx`.
# By default, the offset :math:`a` is zero and the coefficient :math:`b` is one:
default_scaler = Scaler()

# %%
# We can set these coefficient and offset at instantiation:
custom_scaler = Scaler(offset=-1, coefficient=0.5)

# %%
# or use a specific :class:`.Scaler` for that,
# e.g. a :class:`.MinMaxScaler`:
min_max_scaler = MinMaxScaler()

# %%
# or a :class:`.StandardScaler`:
standard_scaler = StandardScaler()

# %%
# In this case,
# the coefficient and offset will be computed from ``data``.

# %%
# Now,
# we fit each scaler from ``data`` and transform these ``data``:
same_data = default_scaler.fit_transform(data)
scaled_data = custom_scaler.fit_transform(data)
min_max_scaled_data = min_max_scaler.fit_transform(data)
standard_scaled_data = standard_scaler.fit_transform(data)

# %%
# We can plot the transformed data versus the original one:
plt.plot(data, default_scaler.fit_transform(data), label="Default scaler")
plt.plot(data, custom_scaler.fit_transform(data), label="Custom scaler")
plt.plot(data, min_max_scaler.fit_transform(data), label="Min-max scaler")
plt.plot(data, standard_scaler.fit_transform(data), label="Standard scaler")
plt.legend()
plt.grid()
plt.show()

# %%
# The specific features of the different scalers are clearly visible.
# In particular,
# the :class:`.MinMaxScaler` projects the data onto the interval :math:`[0,1]`
# as long as this data is included in the fitting interval.
# The :class:`.StandardScaler` guarantees that
# the transformed ``data`` have zero mean and unit variance.
#
# Lastly,
# every scaler can compute the Jacobian,
# e.g.
custom_scaler.compute_jacobian(data)
