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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or
#                      initial documentation
#        :author:  Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import sys
from copy import deepcopy
from os.path import dirname
from os.path import join
from unittest import mock

import pytest
from gemseo.api import create_discipline
from gemseo.wrappers.disc_from_exe import DiscFromExe
from gemseo.wrappers.disc_from_exe import FoldersIter
from gemseo.wrappers.disc_from_exe import parse_key_value_file
from gemseo.wrappers.disc_from_exe import parse_outfile
from gemseo.wrappers.disc_from_exe import parse_template
from gemseo.wrappers.disc_from_exe import Parsers
from numpy import array

from .cfgobj_exe import execute as exec_cfg
from .sum_data import execute as exec_sum

DIRNAME = dirname(__file__)


@pytest.fixture()
def xfail_if_windows_unc_issue(tmp_wd, use_shell=True):
    if str(tmp_wd).startswith("\\\\") and use_shell and sys.platform.startswith("win"):
        pytest.xfail("cmd.exe cannot perform the cd command on a UNC network path.")


@pytest.mark.skipif(
    not sys.platform.startswith("win"), reason="This test is only relevant on Windows."
)
@pytest.mark.xfail(reason="cmd.exe cannot perform the cd command on UNC network path.")
def test_disc_from_exe_network_path_windows():
    """Test that a network location cannot be userd as a workdir under Windows."""

    def _mock_list_out_dir(self) -> list[str]:
        """Mock of _list_out_dir.

        This mock method is needed as an existing workdir is needed by _list_out_dir.
        """
        return []

    with mock.patch.object(DiscFromExe, "_list_out_dirs", new=_mock_list_out_dir):
        create_discipline(
            "DiscFromExe",
            input_template=join(DIRNAME, "input.json.template"),
            output_template=join(DIRNAME, "output.json.template"),
            output_folder_basepath="\\\\my_fake_network_location\\folder\\",
            executable_command="ls",
            input_filename="input.json",
            output_filename="output.json",
            use_shell=True,
        )


@pytest.mark.parametrize("use_shell", [True, False])
def test_disc_from_exe_json(xfail_if_windows_unc_issue, tmp_wd, use_shell):
    sum_path = join(DIRNAME, "sum_data.py")
    exec_cmd = f"python {sum_path} -i input.json -o output.json"

    disc = create_discipline(
        "DiscFromExe",
        input_template=join(DIRNAME, "input.json.template"),
        output_template=join(DIRNAME, "output.json.template"),
        output_folder_basepath=str(tmp_wd),
        executable_command=exec_cmd,
        input_filename="input.json",
        output_filename="output.json",
        use_shell=use_shell,
    )

    indata = {
        "a": array([1.015154]),
        "c": array([3.0015151121254534242424]),
        "b": array([2.001515112125]),
    }
    out = disc.execute(indata)
    assert abs(out["out"] - (indata["a"] + indata["b"] + indata["c"])) < 1e-8

    indata = {"a": array([1]), "c": array([3]), "b": array([2])}
    out = disc.execute(indata)
    assert abs(out["out"] - (indata["a"] + indata["b"] + indata["c"])) < 1e-8


def test_disc_from_exe_cfgobj(xfail_if_windows_unc_issue, tmp_wd):
    sum_path = join(DIRNAME, "cfgobj_exe.py")
    exec_cmd = f"python {sum_path} -i input.cfg -o output.cfg"

    disc = create_discipline(
        "DiscFromExe",
        input_template=join(DIRNAME, "input_template.cfg"),
        output_template=join(DIRNAME, "output_template.cfg"),
        output_folder_basepath=str(tmp_wd),
        executable_command=exec_cmd,
        parse_outfile_method=Parsers.KEY_VALUE_PARSER,
        input_filename="input.cfg",
        output_filename="output.cfg",
    )

    indata = {
        "input 1": array([1.015154]),
        "input 2": array([3.0015151121254534242424]),
        "input 3": array([2.001515112125]),
    }
    out = disc.execute(indata)
    prod_v = indata["input 1"] * indata["input 2"] * indata["input 3"]
    assert abs(out["out 1"] - prod_v) < 1e-8

    indata = {"input 1": array([1]), "input 2": array([3]), "input 3": array([2])}
    out = disc.execute(indata)

    prod_v = indata["input 1"] * indata["input 2"] * indata["input 3"]
    assert abs(out["out 1"] - prod_v) < 1e-8

    disc = create_discipline(
        "DiscFromExe",
        input_template=join(DIRNAME, "input_template.cfg"),
        output_template=join(DIRNAME, "output_template.cfg"),
        output_folder_basepath=str(tmp_wd),
        executable_command=exec_cmd,
        parse_outfile_method=Parsers.KEY_VALUE_PARSER,
        input_filename="input.cfg",
        output_filename="output.cfg",
        folders_iter=FoldersIter.UUID,
    )

    disc.execute(indata)


@pytest.mark.parametrize(
    "folders_iter",
    [
        ("UUID", FoldersIter.UUID),
        (FoldersIter.UUID, FoldersIter.UUID),
        (FoldersIter.NUMBERED, FoldersIter.NUMBERED),
        ("NUMBERED", FoldersIter.NUMBERED),
    ],
)
def test_disc_from_exe_cfgobj_folder_iter_str(
    xfail_if_windows_unc_issue, tmp_wd, folders_iter
):
    sum_path = join(DIRNAME, "cfgobj_exe.py")
    exec_cmd = f"python {sum_path} -i input.cfg -o output.cfg"

    disc = create_discipline(
        "DiscFromExe",
        input_template=join(DIRNAME, "input_template.cfg"),
        output_template=join(DIRNAME, "output_template.cfg"),
        output_folder_basepath=str(tmp_wd),
        executable_command=exec_cmd,
        parse_outfile_method="KEY_VALUE_PARSER",
        input_filename="input.cfg",
        output_filename="output.cfg",
        folders_iter=folders_iter[0],
    )
    assert disc.folders_iter == folders_iter[1]


@pytest.mark.parametrize(
    "parser",
    [
        ("TEMPLATE_PARSER", Parsers.TEMPLATE_PARSER),
        (Parsers.TEMPLATE_PARSER, Parsers.TEMPLATE_PARSER),
        (Parsers.KEY_VALUE_PARSER, Parsers.KEY_VALUE_PARSER),
        ("KEY_VALUE_PARSER", Parsers.KEY_VALUE_PARSER),
        (lambda a, b: zip(a, b), Parsers.CUSTOM_CALLABLE),
    ],
)
def test_disc_from_exe_cfgobj_parser_str(xfail_if_windows_unc_issue, tmp_wd, parser):
    sum_path = join(DIRNAME, "cfgobj_exe.py")
    exec_cmd = f"python {sum_path} -i input.cfg -o output.cfg"

    disc = create_discipline(
        "DiscFromExe",
        input_template=join(DIRNAME, "input_template.cfg"),
        output_template=join(DIRNAME, "output_template.cfg"),
        output_folder_basepath=str(tmp_wd),
        executable_command=exec_cmd,
        parse_outfile_method=parser[0],
        input_filename="input.cfg",
        output_filename="output.cfg",
        folders_iter="UUID",
    )
    assert disc._parse_outfile_method == parser[1]


def test_disc_from_exe_invalid_folder_iter(xfail_if_windows_unc_issue, tmp_wd):
    """Test that a ValueError Exception is raised if an incorrect folder iter is
    provided."""
    sum_path = join(DIRNAME, "cfgobj_exe.py")
    exec_cmd = f"python {sum_path} -i input.cfg -o output.cfg"

    with pytest.raises(
        ValueError, match="wrong_folder_iter is not a valid FoldersIter value."
    ):
        create_discipline(
            "DiscFromExe",
            input_template=join(DIRNAME, "input_template.cfg"),
            output_template=join(DIRNAME, "output_template.cfg"),
            output_folder_basepath=str(tmp_wd),
            executable_command=exec_cmd,
            parse_outfile_method=Parsers.KEY_VALUE_PARSER,
            input_filename="input.cfg",
            output_filename="output.cfg",
            folders_iter="wrong_folder_iter",
        )


def test_exec_cfg(tmp_wd):
    outfile = "out_dummy.cfg"
    infile = join(DIRNAME, "input.cfg")
    exec_cfg(infile, outfile)


def test_exec_json(tmp_wd):
    outfile = "out_dummy.json"
    infile = join(DIRNAME, "input.json")
    exec_sum(infile, outfile)


def test_disc_from_exe_wrong_inputs(tmp_wd):
    sum_path = join(DIRNAME, "cfgobj_exe.py")
    exec_cmd = "python " + sum_path + " -i input.cfg -o output.cfg"

    with pytest.raises(TypeError):
        create_discipline(
            "DiscFromExe",
            input_template=join(DIRNAME, "input_template.cfg"),
            output_template=join(DIRNAME, "output_template.cfg"),
            output_folder_basepath=str(tmp_wd),
            executable_command=exec_cmd,
            parse_outfile_method="ERROR",
            input_filename="input.cfg",
            output_filename="output.cfg",
        )

    with pytest.raises(TypeError):
        create_discipline(
            "DiscFromExe",
            input_template=join(DIRNAME, "input_template.cfg"),
            output_template=join(DIRNAME, "output_template.cfg"),
            output_folder_basepath=str(tmp_wd),
            executable_command=exec_cmd,
            write_input_file_method="ERROR",
            input_filename="input.cfg",
            output_filename="output.cfg",
        )


def test_disc_from_exe_fail_exe(xfail_if_windows_unc_issue, tmp_wd):
    sum_path = join(DIRNAME, "cfgobj_exe_fails.py")
    exec_cmd = "python " + sum_path + " -i input.cfg -o output.cfg -f wrong_len"
    disc = create_discipline(
        "DiscFromExe",
        input_template=join(DIRNAME, "input_template.cfg"),
        output_template=join(DIRNAME, "output_template.cfg"),
        output_folder_basepath=str(tmp_wd),
        executable_command=exec_cmd,
        parse_outfile_method=Parsers.KEY_VALUE_PARSER,
        input_filename="input.cfg",
        output_filename="output.cfg",
    )

    indata = {"input 1": array([1]), "input 2": array([3]), "input 3": array([2])}
    with pytest.raises(ValueError):
        disc.execute(indata)

    disc.reset_statuses_for_run()
    exec_cmd = "python " + sum_path + " -i input.cfg -o output.cfg -f err_code"
    disc.executable_command = exec_cmd
    with pytest.raises(RuntimeError):
        disc.execute(indata)


def test_parse_key_value_file():
    data = parse_key_value_file(None, ["a = 1.0"])
    assert data["a"] == 1.0
    assert len(data) == 1
    data = parse_key_value_file(None, ["a = 1.", "b=2", " c =    4", "d=1e-2"])
    assert data["a"] == 1.0
    assert data["b"] == 2.0
    assert data["c"] == 4.0
    assert data["d"] == 1e-2
    assert len(data) == 4

    data = parse_key_value_file(None, ["a : 1.0"], ":")
    assert data["a"] == 1.0
    assert len(data) == 1

    with pytest.raises(ValueError):
        parse_key_value_file(None, ["a = 1.0 = b"])

    with pytest.raises(ValueError):
        parse_key_value_file(None, ["a = 1.0b"])


def test_parse_outfile():
    with open(join(DIRNAME, "output_template.cfg")) as infile:
        out_template = infile.readlines()

    _, out_pos = parse_template(out_template, False)
    with open(join(DIRNAME, "output.cfg")) as infile:
        output = infile.readlines()
    values = parse_outfile(out_pos, output)

    output_mod = deepcopy(output)
    output_mod[0] = output_mod[0].replace("1.4", "1.4e0")
    values2 = parse_outfile(out_pos, output_mod)
    assert values2 == values

    output_mod = deepcopy(output)
    output_mod[0] = output_mod[0].replace("1.4", "1.400000e0")
    values2 = parse_outfile(out_pos, output_mod)
    assert values2 == values

    output_mod = deepcopy(output)
    del output_mod[-1]
    values2 = parse_outfile(out_pos, output_mod)
    assert values2 != values

    output_mod = deepcopy(output)
    output_mod[-1] = output_mod[-1][:-1] + "1341"
    values2 = parse_outfile(out_pos, output_mod)
    assert values2 != values

    output_mod = deepcopy(output)
    output_mod[0] = output_mod[0][:-1]
    values2 = parse_outfile(out_pos, output_mod)
    assert values2["out 1"] == 1.0
