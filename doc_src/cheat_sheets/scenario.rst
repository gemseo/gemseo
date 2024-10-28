..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

========
Scenario
========

.. code-block:: python

    from gemseo import create_discipline
    from gemseo import create_scenario
    from gemseo import read_design_space

Instantiate an MDO or DOE scenario:

.. code-block:: python

    disc_names = ["disc1", "disc2", "disc3"]
    disciplines = create_discipline(disc_names, "ext_path")
    d_space = read_design_space("file.csv")
    scenario_type = "MDO"  # or 'DOE'
    scenario = create_scenario(
        disciplines,
        formulation="MDF",
        objective_name="obj",
        design_space=d_space,
        name="my_scenario",
        scenario_type=scenario_type,
        **formulation_options,
    )
    scenario.add_constraint("cstr1", "ineq")  # <=0
    scenario.add_constraint("cstr2", "ineq", positive=True)  # >=0
    scenario.add_constraint("cstr3", "ineq", value=1.0)  # <=1
    scenario.add_constraint("cstr4", "eq")  # =0
    scenario.xdsmize()  # Build the XDSM graph to check it.

Execute the scenario:

.. code-block:: python

    scenario.execute(algo_name="SLSQP", max_iter=50, xtol_rel=1e-3)  # if MDO
    scenario.execute(algo_name="LHS", n_samples=30)  # if DOE
    optimum = scenario.get_optimum()

Save the optimisation history:

.. code-block:: python

    optimum_result = scenario.save_optimization_history("file.h5")
