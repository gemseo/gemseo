---
status: draft
description: ""
tags: ['how_to']
search:
  boost: 1
---

<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

<!--
Contributors:
        :author:  Gilberto Ruiz Jimenez
-->

# MDA Settings

The available GEMSEO MDAs require different configuration settings to be instantiated:

``` python
from gemseo.settings.formulations import MDF_Settings
from gemseo.settings.mda import MDAGaussSeidel_Settings

scenario = create_scenario(
    disciplines,
    "y_4",
    design_space,
    maximize_objective=True,
    formulation_settings_model=MDF_Settings(
        main_mda_settings=MDAGaussSeidel_Settings(
            max_mda_iter=50, warm_start=True, linear_solver_tolerance=1e-16
        ),
    ),
)
```
