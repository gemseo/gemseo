..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author:  Gilberto Ruiz Jimenez

.. _algorithm_settings:

Algorithm Settings
==================

The available |g| algorithms require different configuration settings to be executed. There are two ways to pass
algorithm settings, either one by one:

.. code-block:: python

    scenario.execute(algo_name="SLSQP", max_iter=10, ftol_rel=1e-10, ineq_tolerance=2e-3, normalize_design_space=True)

or via their associated Pydantic model:

.. code-block:: python

    from gemseo.settings.opt import SLSQP_Settings

    slsqp_settings = SLSQP_Settings(
        max_iter=10,
        ftol_rel=1e-10,
        ineq_tolerance=2e-3,
        normalize_design_space=True,
    )

    scenario.execute(slsqp_settings)

The advantage of using the Pydantic model directly is that IDEs provide auto completion for Pydantic models. Another
advantage is that the settings are validated at instantiation of the Pydantic model, so in case a setting name or
setting value is not valid, the validation exception will happen before the execution of the scenario, thus saving time.
