# -*- coding: utf-8 -*-
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

from copy import deepcopy
from os import remove
from os.path import dirname, join

import pytest
from numpy import array

from gemseo import SOFTWARE_NAME
from gemseo.api import configure_logger, create_discipline
from gemseo.wrappers.disc_from_exe import (
    OUTPUT_GRAMMAR,
    DiscFromExe,
    parse_key_value_file,
    parse_outfile,
    parse_template,
)

from .cfgobj_exe import execute as exec_cfg
from .sum_data import execute as exec_sum

DIRNAME = dirname(__file__)
configure_logger(SOFTWARE_NAME)


def test_disc_from_exe_json(tmpdir):
    workdir = str(tmpdir)
    sum_path = join(dirname(__file__), "sum_data.py")
    exec_cmd = "python " + sum_path + " -i input.json -o output.json"

    disc = create_discipline(
        "DiscFromExe",
        input_template=join(DIRNAME, "input.json.template"),
        output_template=join(DIRNAME, "output.json.template"),
        output_folder_basepath=workdir,
        executable_command=exec_cmd,
        input_filename="input.json",
        output_filename="output.json",
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


def test_disc_from_exe_cfgobj(tmpdir):
    workdir = str(tmpdir)
    sum_path = join(dirname(__file__), "cfgobj_exe.py")
    exec_cmd = "python " + sum_path + " -i input.cfg -o output.cfg"

    disc = create_discipline(
        "DiscFromExe",
        input_template=join(DIRNAME, "input_template.cfg"),
        output_template=join(DIRNAME, "output_template.cfg"),
        output_folder_basepath=workdir,
        executable_command=exec_cmd,
        parse_outfile_method="KEY_VALUE_PARSER",
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
        output_folder_basepath=workdir,
        executable_command=exec_cmd,
        parse_outfile_method="KEY_VALUE_PARSER",
        input_filename="input.cfg",
        output_filename="output.cfg",
        folders_iter=DiscFromExe.UUID,
    )

    disc.execute(indata)


def test_exec_cfg(tmpdir):
    outfile = join(str(tmpdir), "out_dummy.cfg")
    infile = join(DIRNAME, "input.cfg")
    exec_cfg(infile, outfile)
    remove(outfile)


def test_exec_json(tmpdir):
    outfile = join(str(tmpdir), "out_dummy.json")
    infile = join(DIRNAME, "input.json")
    exec_sum(infile, outfile)
    remove(outfile)


def test_disc_from_exe_wrong_inputs(tmpdir):

    sum_path = join(dirname(__file__), "cfgobj_exe.py")
    exec_cmd = "python " + sum_path + " -i input.cfg -o output.cfg"

    with pytest.raises(TypeError):
        create_discipline(
            "DiscFromExe",
            input_template=join(DIRNAME, "input_template.cfg"),
            output_template=join(DIRNAME, "output_template.cfg"),
            output_folder_basepath=str(tmpdir),
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
            output_folder_basepath=str(tmpdir),
            executable_command=exec_cmd,
            write_input_file_method="ERROR",
            input_filename="input.cfg",
            output_filename="output.cfg",
        )


def test_disc_from_exe_fail_exe(tmpdir):

    sum_path = join(dirname(__file__), "cfgobj_exe_fails.py")
    exec_cmd = "python " + sum_path + " -i input.cfg -o output.cfg -f wrong_len"
    disc = create_discipline(
        "DiscFromExe",
        input_template=join(DIRNAME, "input_template.cfg"),
        output_template=join(DIRNAME, "output_template.cfg"),
        output_folder_basepath=str(tmpdir),
        executable_command=exec_cmd,
        parse_outfile_method="KEY_VALUE_PARSER",
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
    with pytest.raises(ValueError):
        parse_template("template_lines", grammar_type="FAIL")

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
    with open(join(DIRNAME, "output_template.cfg"), "r") as infile:
        out_template = infile.readlines()

    _, out_pos = parse_template(out_template, OUTPUT_GRAMMAR)
    with open(join(DIRNAME, "output.cfg"), "r") as infile:
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
