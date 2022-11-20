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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Newton-Raphson MDA
==================
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import create_discipline
from gemseo.api import create_mda

configure_logger()


#############################################################################
# Define a way to display results
# -------------------------------
def display_result(res, mda_name):
    """Display coupling and output variables in logger.

    @param res: result (dict) of MDA
    @param mda_name: name of the current MDA
    """
    # names of the coupling variables
    coupling_names = [
        "y_11",
        "y_12",
        "y_14",
        "y_21",
        "y_23",
        "y_24",
        "y_31",
        "y_32",
        "y_34",
    ]
    for coupling_var in coupling_names:
        print(
            "{}, coupling variable {}: {}".format(
                mda_name, coupling_var, res[coupling_var]
            ),
        )

    # names of the output variables
    output_names = ["y_1", "y_2", "y_3", "y_4", "g_1", "g_2", "g_3"]
    for output_name in output_names:
        print(
            f"{mda_name}, output variable {output_name}: {res[output_name]}",
        )


#############################################################################
# Create, execute and post-process MDA
# ------------------------------------
# We do not need to specify the inputs, the default inputs
# of the MDA will be used and computed from the
# Default inputs of the disciplines

disciplines = create_discipline(
    [
        "SobieskiStructure",
        "SobieskiPropulsion",
        "SobieskiAerodynamics",
        "SobieskiMission",
    ]
)
mda = create_mda("MDANewtonRaphson", disciplines, relax_factor=0.9)
res = mda.execute()
display_result(res, mda.name)
mda.plot_residual_history(
    n_iterations=10,
    logscale=[1e-8, 10.0],
    save=False,
    show=True,
    fig_size=(10, 2),
)
