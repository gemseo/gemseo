{#-
The fragments are Markdown.
towncrier indents the continuation lines of a fragment by 2 spaces,
which is undone here:
a fragment written as a list is rendered as top-level items,
another one as a single item whose continuation lines are indented by 4 spaces,
as MkDocs requires.
The issue links are appended to the last line of the fragment,
or follow it in their own paragraph when it ends with a sub-list,
so that they do not seem to concern its last sub-item only.
The breaking changes are rendered first in the Changed section,
and the Removed section states that all its items are breaking changes.
#}
{% macro render(entries) %}
{% for text, values in entries.items() %}
{% set text = text.replace("\n  ", "\n") %}
{% set issues = values|join(", ") %}
{% if text[:2] in ("- ", "* ") %}
{{ text }}{% if issues %} {{ issues }}{% endif %}

{% elif issues and ("\n- " in text or "\n* " in text) %}
- {{ text|indent(4) }}

    {{ issues }}
{% else %}
- {{ text|indent(4) }}{% if issues %} {{ issues }}{% endif %}

{% endif %}
{% endfor %}
{% endmacro %}

{% for section in sections %}
{% set categories = sections[section] %}
{% for category, definition in definitions.items() if category != "breaking" %}
{% if category in categories or (category == "changed" and "breaking" in categories) %}
{{ header_prefix }}# {{ definition["name"] }}

{% if category == "changed" and "breaking" in categories %}
{{ header_prefix }}## {{ definitions["breaking"]["name"] }}

{{ render(categories["breaking"]) }}
{% if "changed" in categories %}
{{ header_prefix }}## Other changes

{{ render(categories["changed"]) }}
{% endif %}
{% else %}
{% if category == "removed" %}
All the removals are breaking changes.

{% endif %}
{{ render(categories[category]) }}
{% endif %}
{% endif %}
{% endfor %}
{% endfor %}
