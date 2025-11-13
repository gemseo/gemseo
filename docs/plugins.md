<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Plugins

GEMSEO features can be extended with external Python modules.
All kinds of additional features can be implemented:
disciplines, optimizers, DOE algorithms, formulations, post-processors, surrogate models, etc.

| Name / Repo | Description | Documentation |
|-------------|-------------|---------------|
<!-- markdownlint-disable-next-line MD056 -->
{%- for name, value in gemseo_plugins.items()|sort(attribute='0') %}
<!-- markdownlint-disable-next-line -->
| [{{ name }}](https://gitlab.com/gemseo/dev/{{ name }}) | {{ value }} | [Documentation](https://gemseo.gitlab.io/dev/{{ name }}) |
{%- endfor %}

!!! info "See Also"

    [Extending GEMSEO](extending.md) with external Python modules.

!!! info "See Also"

    Create a new plugin with [copier-gemseo](https://gitlab.com/gemseo/dev/copier-gemseo).
