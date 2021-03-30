..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

.. _correlations:

Correlations
************

A correlation coefficient indicates whether there is a linear
relationship between 2 quantities :math:`x` and :math:`y`, in which case
it equals 1 or -1. It is the normalized covariance between the two
quantities:

.. math::

   R_{xy}=\frac {\sum \limits _{i=1}^n(x_i-{\bar{x}})(y_i-{\bar{y}})}{ns_{x}s_{y}}=
   \frac {\sum \limits _{i=1}^n(x_i-{\bar{x}})(y_i-{\bar{y}})}{\sqrt {\sum
   \limits _{i=1}^n(x_i-{\bar{x}})^{2}\sum \limits _{i=1}^n(y_i-{\bar{y}})^{2}}}

To compute the correlations between all inputs and all outputs as well
as between two outputs, use the API method :meth:`~gemseo.api.execute_post`
with the keyword :code:`“Correlations”` and
additional arguments concerning the type of display (file, screen, both):

.. code::

    scenario.post_process(“Correlations”, coeff_limit=0.85, save=True,
    show=False, n_plots_x=4, n_plots_y=4)

where:

-  ``coeff_limit`` is the absolute threshold for correlation plots. It
   filters the minimum correlation coefficient to be displayed,

-  ``n_plot_x`` and ``n_plot_y`` are the numbers of plots along the
   columns and the rows, respectively.

.. figure:: /_images/postprocessing/DOE_MDF_correlations_1.png
   :alt: Correlation coefficients on the Sobieski use case for the MDF formulation
   :width: 12.00000cm

   Correlation coefficients on the Sobieski use case for the MDF
   formulation

As mentioned earlier, correlation plots highlight the strong correlations between stress
constraints in wing sections : the correlation coefficients belong to
:math:`[0.94766, 0.999286]`.

The aerodynamics constraint ``g_2`` is a polynomial function of
:math:`x_1`: :math:`g\_2=1+0.2\overline{x_1}` with
:math:`\overline{x_1}` the normalized value of :math:`x_1`.
