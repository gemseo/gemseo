{#-
The fragments are Markdown.
towncrier indents the continuation lines of a fragment by 2 spaces,
which is undone here:
a fragment written as a list is rendered as top-level items,
another one as a single item whose continuation lines are indented by 4 spaces,
as MkDocs requires.
The issue links are appended to the last line of the fragment.
#}

{% for section in sections %}
{% for category, definition in definitions.items() if category in sections[section] %}
{{ header_prefix }}# {{ definition["name"] }}

{% for text, values in sections[section][category].items() %}
{% set text = text.replace("\n  ", "\n") %}
{% set issues = values|join(", ") %}
{% if text[:2] in ("- ", "* ") %}
{{ text }}{% if issues %} {{ issues }}{% endif %}

{% else %}
- {{ text|indent(4) }}{% if issues %} {{ issues }}{% endif %}

{% endif %}
{% endfor %}

{% endfor %}
{% endfor %}
