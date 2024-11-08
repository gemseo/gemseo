..
    Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

    This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
    International License. To view a copy of this license, visit
    http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
    Commons, PO Box 1866, Mountain View, CA 94042, USA.

{{ fullname }}

{{ underline }}


.. automodule:: {{ fullname }}


   {% block functions %}

   {% if functions %}


   Functions

   ---------


   {% for item in functions %}


   .. autofunction:: {{ item }}


   .. _sphx_glr_backref_{{fullname}}.{{item}}:


   .. minigallery:: {{fullname}}.{{item}}

       :add-heading:


   {%- endfor %}

   {% endif %}

   {% endblock %}


   {% block classes %}

   {% if classes %}


   Classes

   -------


   {% for item in classes %}

   .. autoclass:: {{ item }}

      :members:


   .. _sphx_glr_backref_{{fullname}}.{{item}}:


   .. minigallery:: {{fullname}}.{{item}}

       :add-heading:


   {%- endfor %}

   {% endif %}

   {% endblock %}


   {% block exceptions %}

   {% if exceptions %}


   Exceptions

   ----------


   .. autosummary::

   {% for item in exceptions %}

      {{ item }}

   {%- endfor %}

   {% endif %}

   {% endblock %}
