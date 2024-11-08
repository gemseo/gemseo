..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

============
Design space
============

.. code-block:: python

    from gemseo import create_design_space
    from gemseo import read_design_space
    from gemseo import write_design_space

Read a design space from a file and handle it:

.. code-block:: python

    design_space = read_design_space("file.csv")
    design_space.filter(["x", "y"])  # Keep x & y variables
    design_space.add_variable("z", lower_bound=-3, upper_bound=2)
    design_space.remove_variable("x")
    print(design_space)  # Pretty table view

Create a design space from scratch and handle it:

.. code-block:: python

    design_space = create_design_space()
    design_space.add_variable("z", size=2, lower_bound=-3, upper_bound=2)

Export a design space to a text or HDF file:

.. code-block:: python

    write_design_space(design_space, "file.csv")
