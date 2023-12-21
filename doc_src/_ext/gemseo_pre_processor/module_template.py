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
from __future__ import annotations

import inspect
import shutil
import sys
from pathlib import Path

MOD_MSG = (
    "<li>"
    "<a href='{}.html'>"
    "<span class='fa-li'><i class='far fa-file-alt'></i></span>{}"
    "</a>"
    "</li>\n"
)

PKG_MSG = (
    "<li><a href='{}.html'>"
    "<span class='fa-li'><i class='far fa-folder'></i></span>{}"
    "</a></li>\n"
)

LINK_MSG = "<a href='{}.html'>{}</a>"


def initialize_file_tree(lst):
    """Initialize the rst file tree where rst files can represent either packages or
    modules.

    :param list(str) lst: list of rst files.
    """
    rev_list = [list(reversed(sub_lst.split(".")[0:-1])) for sub_lst in lst]
    tree = {}
    for item in rev_list:
        curr_tree = tree

        for key in item[::-1]:
            if key not in curr_tree:
                curr_tree[key] = {}
            curr_tree = curr_tree[key]

    return tree


def underline(title, char="="):
    """Underline a title with a character.

    :param str title: title.
    :param str char: underlining character.
    """
    return len(title) * char


def create_tree_file(modules_path, dct, parents, root):
    """Create a rst tree file.

    :param dict dct: dictionary
    :param list(str) parents: parent hierarchy.
    """
    items = sorted(dct.keys())
    parent_path = ".".join(parents)
    parent_rst = Path(f"{parent_path}.rst")
    for index, name in enumerate(items):
        path = f"{parent_path}.{name}"
        path_rst = Path(f"{path}.rst")
        if dct[name]:  # package
            with open(modules_path / parent_rst, "a") as f:
                f.write("      " + PKG_MSG.format(path, name))
                if index + 1 == len(items):
                    f.write("   </ul>\n\n")
                    f.write(f".. automodule:: {parent_path}\n")
                    f.write("   :members:\n")
                    f.write("   :undoc-members:\n")
                    f.write("   :show-inheritance:\n\n")
            with open(modules_path / path_rst, "w") as f:
                f.write(":orphan:\n\n")
                f.write(".. {}:\n\n".format(path.replace(".", "-")))
                gparents = ""
                f.write(".. raw:: html\n\n")
                f.write("   <i class='fa fa-home' id='module-breadcrumb'></i> ")
                for parent in parents:
                    gparents = gparents + "." + parent
                    if parent == root:
                        f.write("   " + LINK_MSG.format(gparents[1:], parent))
                    else:
                        f.write(" / " + LINK_MSG.format(gparents[1:], parent))
                f.write("\n\n")
                f.write(f"{path}\n")
                f.write(f"{underline(path)}\n\n")
                f.write(".. raw:: html\n\n")
                f.write("   <ul class='fa-ul'>\n")
            create_tree_file(modules_path, dct[name], [*parents, name], root)
        else:  # module
            with open(modules_path / parent_rst, "a") as f:
                f.write("      " + MOD_MSG.format(path, name))
                if index + 1 == len(items):
                    f.write("   </ul>\n\n")
                    f.write(f".. automodule:: {parent_path}\n")
                    f.write("   :members:\n")
                    f.write("   :undoc-members:\n")
                    f.write("   :show-inheritance:\n\n")
            with open(modules_path / f"tmp_{path_rst}", "w") as f:
                f.write(":orphan:\n\n")
                f.write(f".. _{path}:\n\n")
                gparents = ""
                f.write(".. raw:: html\n\n")
                f.write("   <i class='fa fa-home' id='module-breadcrumb'></i> ")
                for parent in parents:
                    gparents = gparents + "." + parent
                    if parent == root:
                        f.write("   " + LINK_MSG.format(gparents[1:], parent))
                    else:
                        f.write(" / " + LINK_MSG.format(gparents[1:], parent))
                f.write("\n\n")
                is_dataset = path.startswith("gemseo.dataset")
                if not is_dataset:
                    f.write(".. raw:: html\n\n")
                    f.write(
                        "   <p align='right'; style='position: sticky; top: 50px;'>"
                        f"<a class='btn sk-landing-btn mb-1' href='{path}_.html'>"
                        "Hide inherited members"
                        "</a>"
                        "</p>"
                    )
                    f.write("\n\n")
                with open(modules_path / path_rst) as fr:
                    title = fr.readline()
                title = title.split(".")[-1]
                f.write(title)
                f.write(f"{underline(title)}\n")
                with open(modules_path / path_rst) as fr:
                    lines = fr.readlines()[2:]
                    for line in lines:
                        f.write(line)

                    if is_dataset:
                        f.write("   :no-inherited-members:\n")
                    else:
                        f.write("   :inherited-members:\n")

                if path != "gemseo.utils.pytest_conftest":
                    obj_names = [
                        obj_name
                        for (obj_name, _) in inspect.getmembers(
                            sys.modules[path],
                            lambda member: (
                                inspect.isclass(member) or inspect.isfunction(member)
                            )
                            and member.__module__ == path,  # noqa: B023
                        )
                    ]
                    for obj_name in obj_names:
                        f.write("\n")
                        f.write(f".. minigallery:: {path}.{obj_name}\n")
                        f.write(f"   :add-heading: Examples using {obj_name}\n")

            (modules_path / path_rst).unlink()
            old_path = modules_path / f"tmp_{path}.rst"
            path_with_inherited_members = modules_path / path_rst
            old_path.rename(path_with_inherited_members)
            path_without_no_inherited_members = modules_path / f"{path}_.rst"
            shutil.copy(path_with_inherited_members, path_without_no_inherited_members)

            with path_with_inherited_members.open("r") as f:
                data = f.read()

            if not is_dataset:
                data = data.replace(
                    "   :inherited-members:\n",
                    "   :no-inherited-members:\n   :noindex:\n",
                )
                data = data.replace("_.html'>Hide", ".html'>Show")
                with path_without_no_inherited_members.open("w") as f:
                    f.write(data)


def main(modules_path, name):
    tree = initialize_file_tree([
        f.name
        for f in modules_path.iterdir()
        if f.is_file() and f.name != "modules.rst"
    ])

    file_path = modules_path / Path(name).with_suffix(".rst")
    with open(file_path, "w") as f:
        f.write(f".. _{name}:\n\n")
        f.write(".. raw:: html\n\n")
        f.write("   <i class='fa fa-home'></i> \n\n")
        f.write(f"{name}\n")
        f.write(f"{underline(name)}\n\n")

    sub_tree = tree.get(name)
    if sub_tree:
        with open(file_path, "a") as f:
            f.write(".. raw:: html\n\n")
            f.write("   <ul class='fa-ul'>\n")

        create_tree_file(modules_path, sub_tree, [name], name)
    else:
        with open(file_path, "a") as f:
            f.write(f".. automodule:: {name}\n")
            f.write("   :noindex:\n\n")
