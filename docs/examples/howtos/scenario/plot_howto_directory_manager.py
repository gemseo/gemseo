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
#                           documentation
#        :author: Gilberto RUIZ JIMENEZ
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""# Using the DirectoryManager
## Problem
When disciplines create and read files at run time, the number of files grows quickly
and it becomes hard to tell which file belongs to which discipline or iteration.

## Solution
Enable the ``DirectoryManager`` via ``configuration.directory_manager``:
it organizes run-time files into a directory hierarchy that mirrors the scenario workflow
and lets you apply cleanup policies to control disk usage.

## Step-by-step guide
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from numpy import array
from numpy import ones

from gemseo import configuration
from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo.algos.opt.nlopt.settings.nlopt_cobyla_settings import NLOPT_COBYLA_Settings
from gemseo.core.discipline import Discipline
from gemseo.formulations.mdf_settings import MDF_Settings
from gemseo.utils._directory_manager.settings import CleanUpPolicy
from gemseo.utils._directory_manager.settings import MDACleanUpPolicy

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping
# %%
# !!! warning
#     The DirectoryManager is an experimental feature. It has been tested on toy
#     problems and certain real-world applications, but it may not work as expected in
#     complex workflow setups. If you find a bug, do not hesitate to
#     [create an issue on Gitlab](https://gitlab.com/gemseo/dev/gemseo/-/work_items).
#
# ### 1. Activate the directory manager
#
# Use the gemseo [configuration][concept-global-configuration] to enable the directory manager.
# Call this as early as possible, it sets global state used by all subsequent objects.
# If your script includes an `if __name__ == "__main__":` statement, the configuration
# must be set outside of it so that it is taken into account all the time.
configuration.directory_manager.enable = True

# %%
# ### 2. Set the scenario cleanup policy
#
# Available policies:
#
# - [CleanUpPolicy.KEEP_ALL][gemseo.utils._directory_manager.settings.CleanUpPolicy.KEEP_ALL]: keep all directories
# (default);
# - [CleanUpPolicy.KEEP_LAST_ONLY][gemseo.utils._directory_manager.settings.CleanUpPolicy.KEEP_LAST_ONLY]: keep only
# the last iteration directories;
# - [CleanUpPolicy.KEEP_SOLUTION_ONLY][gemseo.utils._directory_manager.settings.CleanUpPolicy.KEEP_SOLUTION_ONLY]: keep
# only the solution directory;
# - [CleanUpPolicy.KEEP_BASELINE_AND_SOLUTION][gemseo.utils._directory_manager.settings.CleanUpPolicy.KEEP_BASELINE_AND_SOLUTION]:
# keep the baseline and solution
#   directories.
configuration.directory_manager.clean_up_policy = CleanUpPolicy.KEEP_ALL

# %%
# ### 3. Set the MDA cleanup policy
#
# Available policies:
#
# - [MDACleanUpPolicy.KEEP_ALL][gemseo.utils._directory_manager.settings.MDACleanUpPolicy.KEEP_ALL]: keep all
# directories (default);
# - [MDACleanUpPolicy.KEEP_LAST_ONLY][gemseo.utils._directory_manager.settings.MDACleanUpPolicy.KEEP_LAST_ONLY]: keep
# only the last iteration directories.
configuration.directory_manager.mda_clean_up_policy = MDACleanUpPolicy.KEEP_ALL

# %%
# ### 4. Set the root path
#
# Defaults to the current directory; override to redirect the entire directory structure.
# In this example, we create a temporary working directory and set it as the execution
# root path.
temp_dir = tempfile.TemporaryDirectory()
configuration.directory_manager.execution_root_path = Path(temp_dir.name) / "example"

# %%
# ### 5. Configure optional outputs
#
# ``save_history_backup`` writes a per-execution copy of the scenario history to the
# disk — useful for nested scenarios (e.g. BiLevel) but slows the execution.
# ``save_mda_residuals`` saves a residual plot for every MDA run.
configuration.directory_manager.save_history_backup = False
configuration.directory_manager.save_mda_residuals = False


# %%
# ### 6. Define a file-writing discipline
#
# We define below a custom ``Sellar1`` discipline that writes its output to ``y_1.txt``
# at each call, representing a discipline that produces files at run time.


class Sellar1(Discipline):
    def __init__(self) -> None:
        super().__init__()
        self.input_grammar.update_from_names(["x_local", "x_shared", "y_2"])
        self.output_grammar.update_from_names(["y_1"])
        self.default_input_data = {
            "x_local": ones(1),
            "x_shared": array([4.0, 3.0]),
            "y_2": ones(1),
        }

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        x_local = input_data["x_local"]
        x_shared = input_data["x_shared"]
        y_2 = input_data["y_2"]
        y_1 = array([
            (x_shared[0] ** 2 + x_shared[1] + x_local[0] - 0.2 * y_2[0]) ** 0.5
        ])
        Path("y_1.txt").write_text(str(y_1))
        return {"y_1": y_1}


# %%
# ### 7. Instantiate the disciplines
disciplines = create_discipline(["Sellar2", "SellarSystem"])
disciplines.append(Sellar1())

# %%
# ### 8. Define the design space
design_space = create_design_space()
design_space.add_variable("x_local", lower_bound=0.0, upper_bound=10.0, value=ones(1))
design_space.add_variable(
    "x_shared",
    2,
    lower_bound=(-10, 0.0),
    upper_bound=(10.0, 10.0),
    value=array([4.0, 3.0]),
)

# %%
# ### 9. Create the MDO scenario
scenario = create_scenario(
    disciplines, "obj", design_space, formulation_settings_model=MDF_Settings()
)

# %%
# ### 10. Add constraints
scenario.add_constraint("c_1", constraint_type="ineq")
scenario.add_constraint("c_2", constraint_type="ineq")

# %%
# ### 11. Execute the scenario
cobyla_settings = NLOPT_COBYLA_Settings(max_iter=5)
scenario.execute(cobyla_settings)

# %%
# ## Summary
#
# Enabling ``configuration.directory_manager`` automatically organizes all run-time
# files into a directory hierarchy that mirrors the scenario workflow.
# ``clean_up_policy`` and ``mda_clean_up_policy`` control how much of that hierarchy
# is retained after each run, keeping disk usage predictable across many iterations.
#
# ## One step further
#
# Try ``CleanUpPolicy.KEEP_SOLUTION_ONLY`` to minimize disk usage,
# or use a BiLevel formulation to observe how the directory structure adapts
# to nested scenarios.
