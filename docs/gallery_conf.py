# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Configuration of mkdocs-gallery.

All the directories that must be run are gathered into 2 categories:
*tutorials* and *how-tos*.

A third category has been added (*bulk*) to gather all the examples that have not been modified yet.
"""

from __future__ import annotations

from pathlib import Path

from mkdocs_gallery.gen_gallery import DEFAULT_GALLERY_CONF
from mkdocs_gallery.sorting import _SortKey

file_dir_path = Path(__file__).parent
example_dir_name = "examples"

# TODO: find a way to put this into _docs
examples_dir = file_dir_path / example_dir_name

_LEAKY_EXAMPLES = frozenset({"plot_howto_directory_manager.py"})
"""Examples that leak global state and must run after every other example."""


def _has_leaky_example(subdir: Path) -> bool:
    return any((subdir / name).is_file() for name in _LEAKY_EXAMPLES)


examples_subdirs = []
for category_name in ["bulk", "howtos", "tutorials"]:
    directory_path = examples_dir / category_name
    examples_subdirs += [
        subdir
        for subdir in directory_path.iterdir()
        if subdir.is_dir() and (subdir / "README.md").is_file()
    ]

# Push subsections containing leaky examples to the end so they do not pollute
# later subsections via global state.
examples_subdirs.sort(key=_has_leaky_example)


def _patch_gallery():
    # To get the "reset_modules" to work,
    # we have to hard code _reset_dict similarly to what
    # is already built in it.
    import sys

    from mkdocs_gallery.scrapers import _reset_dict

    sys.path.append(str(file_dir_path / "_scripts"))
    import gallery_logging
    import gallery_style

    _reset_dict["gallery_logging.reset_logging"] = gallery_logging.reset_logging
    _reset_dict["gallery_style.reset_style"] = gallery_style.reset_style


def _patch_py_source_parser_for_py314():
    # mkdocs-gallery uses removed `ast.Str` and `Constant.s`. Replace
    # `_get_docstring_and_rest` with a Constant-based equivalent.
    import ast
    import platform
    import tokenize
    from io import BytesIO

    from mkdocs_gallery import py_source_parser
    from mkdocs_gallery.errors import ExtensionError
    from packaging.version import parse as parse_version

    def _get_docstring_and_rest(file):
        node, content = py_source_parser.parse_source_file(file)
        if node is None:
            return py_source_parser.SYNTAX_ERROR_DOCSTRING, content, 1, node
        if not isinstance(node, ast.Module):
            msg = (
                "This function only supports modules. "
                f"You provided {node.__class__.__name__}"
            )
            raise ExtensionError(msg)
        is_str_const = (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        )
        if not is_str_const:
            msg = (
                f'Could not find docstring in file "{file}". '
                "A docstring is required by mkdocs-gallery "
                'unless the file is ignored by "ignore_pattern"'
            )
            raise ExtensionError(msg)
        if parse_version(platform.python_version()) >= parse_version("3.7"):
            docstring = ast.get_docstring(node)
            assert docstring is not None
            raw = node.body[0].value.value
            if len(raw) and raw[0] == "\n":
                docstring = "\n" + docstring
            ts = tokenize.tokenize(BytesIO(content.encode()).readline)
            for tk in ts:
                if tk.exact_type == 3:
                    lineno, _ = tk.end
                    break
            else:
                lineno = 0
        else:
            docstring_node = node.body[0]
            docstring = docstring_node.value.value
            lineno = docstring_node.lineno
        rest = "\n".join(content.split("\n")[lineno:])
        lineno += 1
        return docstring, rest, lineno, node

    py_source_parser._get_docstring_and_rest = _get_docstring_and_rest


def _patch_intro_extraction():
    # mkdocs-gallery uses the docstring's second paragraph as the intro of an
    # example, but the GEMSEO docstring convention makes it a "## Problem"
    # heading. Use the first paragraph left with prose once its heading lines
    # are removed, and truncate at 200 displayed characters rather than 95,
    # since the intro is now shown next to the title instead of in a tooltip.
    #
    # `mkdocs_gallery.sorting` imports `extract_intro_and_title` by value too
    # but is deliberately left alone: it only uses it in `ExampleTitleSortKey`,
    # while the `within_subsection_order` below is `_LeakyLastSortKey`.
    import re

    from mkdocs_gallery import gen_single
    from mkdocs_gallery.errors import ExtensionError

    max_intro_length = 200
    heading_line = re.compile(r"^\s*#{1,6}\s")
    # A [label] whose ][ref] or ](url) part has been cut off.
    dangling_link = re.compile(r"\[[^\]]*\]\s*[\[(][^\])]*$")
    # A [label][ref] or [label](url) link, including one cut off by a
    # truncation, whose target is not part of what the reader sees.
    link = re.compile(r"\[([^\]]+)\](\[[^\]]*\]?|\([^)]*\)?)")

    def _remove_headings(paragraph):
        # Remove the heading lines instead of the paragraphs starting with a
        # heading: a heading with no blank line below it keeps the prose that
        # follows it in its own paragraph.
        return "\n".join(
            line for line in paragraph.splitlines() if not heading_line.match(line)
        ).strip()

    def _displayed_length(text):
        return len(link.sub(r"\1", text))

    def _truncate(intro):
        if _displayed_length(intro) <= max_intro_length:
            return intro

        # The limit applies to the text as displayed: the reference target of a
        # cross-reference is often longer than its label, and counting it would
        # cut the prose of a link-heavy intro down to a fraction of the others.
        limit = max_intro_length
        while (
            limit < len(intro) and _displayed_length(intro[:limit]) < max_intro_length
        ):
            limit += 1

        truncated = intro[:limit].rsplit(" ", 1)[0]
        while True:
            # Cutting inside a $...$ formula, a `code` span or a [label][ref]
            # link leaves a delimiter that the Markdown renderer cannot pair,
            # and an unresolved cross-reference fails a strict build, hence the
            # back off to the start of whatever the cut left open. Parentheses
            # are not tracked, as prose uses them on their own.
            cuts = [
                truncated.rfind(delimiter)
                for delimiter in ("$", "`")
                if truncated.count(delimiter) % 2
            ]
            opening_bracket = truncated.rfind("[")
            if opening_bracket > truncated.rfind("]") or dangling_link.search(
                truncated
            ):
                cuts.append(opening_bracket)

            if not cuts:
                return truncated + "..."

            truncated = truncated[: min(cuts)].rstrip()

    def extract_intro_and_title(docstring, script):
        paragraphs = gen_single.extract_paragraphs(docstring)
        if len(paragraphs) == 0:
            msg = (
                "Example docstring should have a header for the example title. "
                f"Please check the example file:\n {script.script_file}\n"
            )
            raise ExtensionError(msg)

        title_paragraph = paragraphs[0]
        match = gen_single.FIRST_NON_MARKER_WITHOUT_HASH.search(title_paragraph)
        if match is None:
            msg = f"Could not find a title in first paragraph:\n{title_paragraph}"
            raise ExtensionError(msg)

        title = match.group(2).strip()
        # Fall back to no intro at all instead of to the title, which upstream
        # can afford because the intro only reaches a tooltip; here it would be
        # rendered as "Title — Title".
        intro_paragraph = next(filter(None, map(_remove_headings, paragraphs)), "")
        intro = gen_single._sanitize_md(intro_paragraph.replace("\n", " "))
        return title, _truncate(intro)

    gen_single.extract_intro_and_title = extract_intro_and_title


def _patch_thumbnail_div():
    # Render the gallery index pages as lists of entries made of the linked
    # title of the example followed by its intro, with a small thumbnail on
    # the left when the example produced a figure; the entries relying on the
    # default thumbnail (the GEMSEO monogram) get no image at all.
    # Styled by docs/assets/css/gallery.css.
    import hashlib
    import tempfile
    from html import escape
    from pathlib import Path

    from mkdocs_gallery import backreferences
    from mkdocs_gallery import gen_single
    from mkdocs_gallery import glr_path_static
    from mkdocs_gallery.errors import ExtensionError
    from mkdocs_gallery.utils import rescale_image

    default_thumb_hashes = {}

    def _get_default_thumb_hash(gallery_conf):
        # The default thumbnail file is rescaled to thumbnail_size like any
        # figure, so hashing the same rescaling of the default thumbnail
        # identifies the figure-less examples byte-wise, even for thumbnails
        # cached by a previous build. This assumes that the default thumbnail
        # is a raster, since mkdocs-gallery copies an `.svg` or a `.gif` file
        # instead of rescaling it, and that "thumbnails" is not in
        # `compress_images`, since optipng would then rewrite the bytes of the
        # thumbnails but not the ones hashed here.
        default_thumb_file = gallery_conf.get("default_thumb_file")
        if default_thumb_file is None:
            default_thumb_file = Path(glr_path_static()) / "no_image.png"

        thumbnail_size = gallery_conf["thumbnail_size"]
        key = (str(default_thumb_file), tuple(thumbnail_size))
        if key not in default_thumb_hashes:
            with tempfile.TemporaryDirectory() as directory:
                thumb_path = Path(directory) / "default_thumb.png"
                rescale_image(Path(default_thumb_file), thumb_path, *thumbnail_size)
                digest = hashlib.md5(thumb_path.read_bytes()).hexdigest()

            default_thumb_hashes[key] = digest

        return default_thumb_hashes[key]

    def _thumbnail_div(script_results, is_backref=False, check=True):
        if check and not script_results.thumb.exists():
            msg = (
                "Could not find internal mkdocs-gallery thumbnail file:\n"
                f"{script_results.thumb}"
            )
            raise ExtensionError(msg)

        example_html = script_results.script.md_file_rel_root_gallery.with_suffix(
            ""
        ).as_posix()
        thumb_hash = hashlib.md5(script_results.thumb.read_bytes()).hexdigest()
        if thumb_hash == _get_default_thumb_hash(script_results.script.gallery_conf):
            thumb_link = ""
        else:
            thumbnail = script_results.thumb_rel_root_gallery.as_posix()
            # The image has no accessible name and the link duplicates the one
            # on the title below, so hide it from the assistive technologies
            # instead of letting them read its URL out.
            thumb_link = (
                f'<a href="{example_html}" aria-hidden="true" tabindex="-1">'
                f'<img src="{thumbnail}" alt="" /></a>'
            )

        title = escape(script_results.script.title)
        intro = script_results.intro
        intro_span = (
            f'<span class="gallery-item-intro"> — {intro}</span>' if intro else ""
        )
        # `markdown="span"` lets md_in_html render the Markdown of the intro:
        # its maths, which pymdownx.arithmatex must wrap in an .arithmatex
        # element for MathJax to be loaded at all, but also its emphasis, its
        # inline code and its cross-references. The lines must not be indented,
        # as four spaces would turn them into a code block.
        return f"""
<div class="gallery-item" markdown="1">
<div class="gallery-item-thumb">{thumb_link}</div>
<p class="gallery-item-text" markdown="span">
<a class="reference internal" href="{example_html}">{title}</a>{intro_span}
</p>
</div>
"""

    backreferences._thumbnail_div = _thumbnail_div
    gen_single._thumbnail_div = _thumbnail_div


_patch_gallery()
_patch_py_source_parser_for_py314()
_patch_intro_extraction()
_patch_thumbnail_div()

examples_dir_relative = [
    str(subdir.relative_to(file_dir_path)) for subdir in examples_subdirs
]


def insert_generated_in_path(path: Path) -> Path:
    """Insert the `generated` directory just after the `docs` directory.

    Args:
        path: The path within the `generated` directory must be added.

    Returns:
        The path containing the `generated` directory.
    """
    parts = list(path.parts)
    idx = parts.index("docs") + 1
    parts.insert(idx, "generated")
    return Path(*parts)


class _LeakyLastSortKey(_SortKey):
    """Sort examples by file name, pushing leaky examples to the end."""

    def __call__(self, file: Path) -> tuple[bool, str]:
        return file.name in _LEAKY_EXAMPLES, file.name


conf = {
    "examples_dirs": examples_subdirs,
    "gallery_dirs": [insert_generated_in_path(subdir) for subdir in examples_subdirs],
    # As a precaution, keep the already defined reset modules.
    # `gallery_style.reset_style` must come after the built-in "matplotlib" one,
    # which resets the rcParams to their defaults.
    "reset_modules": DEFAULT_GALLERY_CONF["reset_modules"]
    + ("gallery_logging.reset_logging", "gallery_style.reset_style"),
    "within_subsection_order": _LeakyLastSortKey,
}
