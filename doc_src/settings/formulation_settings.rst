..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author:  Gilberto Ruiz Jimenez

.. _formulation_settings:

Formulation Settings
====================

The available |g| formulations require different configuration settings to be instantiated. There are two ways to pass
formulation settings, either one by one:

.. code-block:: python

    scenario = create_scenario(
        disciplines,
        "y_4",
        design_space,
        maximize_objective=True,
        **formulation_settings,
        formulation_name="MDF",
        main_mda_name="MDAGaussSeidel",
        main_mda_settings={
            "max_mda_iter": 50,
            "warm_start": True,
            "linear_solver_tolerance": 1e-14,
        },
    )

or via their associated Pydantic model:

.. code-block:: python

    from gemseo.settings.formulations import MDF_Settings

    scenario = create_scenario(
        disciplines,
        "y_4",
        design_space,
        maximize_objective=True,
        formulation_settings_model=MDF_Settings(
            main_mda_name="MDAGaussSeidel",
            main_mda_settings={
                "max_mda_iter": 50,
                "warm_start": True,
                "linear_solver_tolerance": 1e-14,
            },
        ),
    )

The advantage of using the Pydantic model directly is that IDEs provide auto completion for Pydantic models. Another
advantage is that the settings are validated at instantiation of the Pydantic model, so in case a setting name or
setting value is not valid, the validation exception will happen before the execution of the scenario, thus saving time.
