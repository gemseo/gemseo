{#-
The fragments are Markdown.
A release is rendered as one flat list of items,
each of them starting with its label in bold, e.g. `- **Added** ...`.
The label is the name of the type of the fragment, set in `towncrier.toml`.
The items are sorted by label, in the order of `labels` below,
and by the order of towncrier within a label.
All the removals are breaking changes,
which the changelog page of the documentation states in its introduction.
towncrier indents the continuation lines of a fragment by 2 spaces,
which is undone here:
a fragment written as a list is rendered as top-level items,
each of them starting with the label,
another one as a single item whose continuation lines are indented by 4 spaces,
as MkDocs requires.
The issue links are appended to the last line of the fragment,
or follow it in their own paragraph when it ends with a sub-list,
so that they do not seem to concern its last sub-item only.
#}
{% set labels = ["breaking", "removed", "deprecated", "added", "changed", "fixed"] %}
{% macro render(label, entries) %}
{% for text, values in entries.items() %}
{% set text = text.replace("\n  ", "\n") %}
{% set issues = values|join(", ") %}
{% if text[:2] in ("- ", "* ") %}
{{ text[:2] }}**{{ label }}** {{ text[2:]|replace("\n- ", "\n- **" ~ label ~ "** ")|replace("\n* ", "\n* **" ~ label ~ "** ") }}{% if issues %} {{ issues }}{% endif %}

{% elif issues and ("\n- " in text or "\n* " in text) %}
- **{{ label }}** {{ text|indent(4) }}

    {{ issues }}
{% else %}
- **{{ label }}** {{ text|indent(4) }}{% if issues %} {{ issues }}{% endif %}

{% endif %}
{% endfor %}
{% endmacro %}

{% for section in sections %}
{% set categories = sections[section] %}
{% for category in labels if category in categories %}
{{ render(definitions[category]["name"], categories[category]) }}
{% endfor %}
{% endfor %}
