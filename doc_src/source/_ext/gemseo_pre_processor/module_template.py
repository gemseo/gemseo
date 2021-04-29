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

from pathlib import Path

NAME = "gemseo"

MOD_MSG = (
    "<li><a href='{}.html'>"
    "<span class='fa-li'><i class='far fa-file-alt'></i></span>{}"
    "</a> - <i><a href='{}.html'>source</a></i></li>\n"
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


def create_tree_file(modules_path, dct, parents):
    """Create a rst tree file.

    :param dict dct: dictionary
    :param list(str) parents: parent hierarchy.
    """
    items = sorted(dct.keys())
    parent_path = ".".join(parents)
    parent_rst = Path("{}.rst".format(parent_path))
    for index, name in enumerate(items):
        path = "{}.{}".format(parent_path, name)
        path_rst = Path("{}.rst".format(path))
        if dct[name]:  # package
            with open(modules_path / parent_rst, "a") as f:
                f.write("      " + PKG_MSG.format(path, name))
                if index + 1 == len(items):
                    f.write("   </ul>\n")
                    f.write(".. automodule:: {}\n".format(parent_path))
                    f.write("   :noindex:\n\n")
            with open(modules_path / path_rst, "w") as f:
                f.write(":orphan:\n\n")
                f.write(".. {}:\n\n".format(path.replace(".", "-")))
                gparents = ""
                f.write(".. raw:: html\n\n")
                f.write("   <i class='fa fa-home'></i> ")
                for parent in parents:
                    gparents = gparents + "." + parent
                    if parent == NAME:
                        f.write("   " + LINK_MSG.format(gparents[1:], parent))
                    else:
                        f.write(" / " + LINK_MSG.format(gparents[1:], parent))
                f.write("\n\n")
                f.write("{}\n".format(path))
                f.write("{}\n\n".format(underline(path)))
                f.write(".. raw:: html\n\n")
                f.write("   <ul class='fa-ul'>\n")
            create_tree_file(modules_path, dct[name], parents + [name])
        else:  # module
            with open(modules_path / parent_rst, "a") as f:
                src_path = path.replace(".", "/")
                f.write("      " + MOD_MSG.format(path, name, src_path))
                if index + 1 == len(items):
                    f.write("   </ul>\n")
                    f.write(".. automodule:: {}\n".format(parent_path))
                    f.write("   :noindex:\n\n")
            with open(modules_path / "tree_{}".format(path_rst), "w") as f:
                f.write(":orphan:\n\n")
                f.write(".. _tree-{}:\n\n".format(path.replace(".", "-")))
                gparents = ""
                f.write(".. raw:: html\n\n")
                f.write("   <i class='fa fa-home'></i> ")
                for parent in parents:
                    gparents = gparents + "." + parent
                    if parent == NAME:
                        f.write("   " + LINK_MSG.format(gparents[1:], parent))
                    else:
                        f.write(" / " + LINK_MSG.format(gparents[1:], parent))
                f.write("\n\n")
                with open(modules_path / path_rst, "r") as fr:
                    title = fr.readline()
                title = title.split(".")[-1]
                f.write(title)
                f.write("{}\n".format(underline(title)))
                with open(modules_path / path_rst, "r") as fr:
                    lines = fr.readlines()[2:]
                    for line in lines:
                        f.write(line)
            (modules_path / path_rst).unlink()
            old_tree_path = modules_path / "tree_{}.rst".format(path)
            new_tree_path = modules_path / path_rst
            old_tree_path.rename(new_tree_path)


def main(modules_path):
    lst = [
        f.name
        for f in modules_path.iterdir()
        if f.is_file() and f.name != "modules.rst"
    ]

    with open(modules_path / Path(NAME).with_suffix(".rst"), "w") as f:
        f.write(".. _gemseo:\n\n")
        f.write(".. raw:: html\n\n")
        f.write("   <i class='fa fa-home'></i> \n\n")
        title = "Package tree structure"
        f.write("{}\n".format(title))
        f.write("{}\n\n".format(underline(title)))
        f.write(".. raw:: html\n\n")
        f.write("   <ul class='fa-ul'>\n")

    tree = initialize_file_tree(lst)
    create_tree_file(modules_path, tree[NAME], [NAME])
