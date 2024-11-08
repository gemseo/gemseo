..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

==============
New discipline
==============

.. code-block:: python

    from numpy import array
    from numpy import atleast_2d

    from gemseo import Discipline

Create a new discipline from scratch:

.. code-block:: python

    class NewDiscipline(Discipline):
        def __init__(self):
            super().__init__()
            self.input_grammar.update_from_names(["x", "z"])
            self.output_grammar.update_from_names(["f"])
            self.default_input_data = {"x": array([0.0]), "z": array([0.0])}

        def _run(self, input_data):
            x = input_data["x"]
            z = input_data["z"]
            f = array([x[0] * z[0]])
            g = array([x[0] * (z[0] + 1.0) ** 2])
            return {"f": f, "g": g}

        def _compute_jacobian(
            self,
            input_names: Iterable[str] = (),
            output_names: Iterable[str] = (),
        ) -> None:
            x = self.local_data["x"]
            z = self.local_data["z"]
            dfdx = z
            dfdz = x
            dgdx = array([(z[0] + 1.0) ** 2])
            dgdz = array([2 * x[0] * z[0] * (z[0] + 1.0)])
            self.jac = {}
            df = self.jac["f"] = {}
            df["x"] = atleast_2d(dfdx)
            df["z"] = atleast_2d(dfdz)
            dg = self.jac["g"] = {}
            dg["x"] = atleast_2d(dgdx)
            dg["z"] = atleast_2d(dgdz)
