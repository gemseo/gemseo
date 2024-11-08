..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author:  Gilberto Ruiz Jimenez

.. _post_processor_settings:

Post-processor Settings
=======================

The available |g| post-processors require different configuration settings to be executed. There are two ways to pass
post-processor settings, either one by one:

.. code-block:: python

    scenario.post_process(
        post_name="BasicHistory", variable_names=["x_shared"], save=False, show=True
    )

or via their associated Pydantic model:

.. code-block:: python

    from gemseo.settings.post import BasicHistory_Settings

    scenario.post_process(
        settings_model=BasicHistory_Settings(variable_names=["x_shared"], save=False, show=True)
    )

The advantage of using the Pydantic model directly is that IDEs provide auto completion for Pydantic models. Another
advantage is that the settings are validated at instantiation of the Pydantic model, so in case a setting name or
setting value is not valid, the validation exception will happen before the execution of the post-processor, thus saving time.
