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

from copy import deepcopy
from os.path import dirname
from os.path import join
from subprocess import CalledProcessError

import pytest
from numpy import array

from gemseo import create_discipline
from gemseo import create_scenario
from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.wrappers.disc_from_exe import DiscFromExe
from gemseo.disciplines.wrappers.disc_from_exe import Parser
from gemseo.disciplines.wrappers.disc_from_exe import parse_key_value_file
from gemseo.disciplines.wrappers.disc_from_exe import parse_outfile
from gemseo.disciplines.wrappers.disc_from_exe import parse_template
from gemseo.utils.directory_creator import DirectoryNamingMethod

from .cfgobj_exe import execute as exec_cfg
from .sum_data import execute as exec_sum

DIRNAME = dirname(__file__)


def test_disc_from_exe_json(tmp_wd) -> None:
    sum_path = join(DIRNAME, "sum_data.py")
    exec_cmd = f"python {sum_path} -i input.json -o output.json"

    disc: DiscFromExe = create_discipline(
        "DiscFromExe",
        input_template=join(DIRNAME, "input.json.template"),
        output_template=join(DIRNAME, "output.json.template"),
        root_directory=tmp_wd,
        command_line=exec_cmd,
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

    indata = {"a": array([1.0]), "c": array([3.0]), "b": array([2.0])}
    out = disc.execute(indata)
    assert abs(out["out"] - (indata["a"] + indata["b"] + indata["c"])) < 1e-8
    disc.set_jacobian_approximation(jac_approx_n_processes=2)
    out_jac = disc.linearize(indata, compute_all_jacobians=True)
    assert abs(out_jac["out"]["a"] - 1) < 1e-8


def test_disc_from_exe_cfgobj(tmp_wd) -> None:
    sum_path = join(DIRNAME, "cfgobj_exe.py")
    exec_cmd = f"python {sum_path} -i input.cfg -o output.cfg"

    disc = create_discipline(
        "DiscFromExe",
        input_template=join(DIRNAME, "input_template.cfg"),
        output_template=join(DIRNAME, "output_template.cfg"),
        root_directory=tmp_wd,
        command_line=exec_cmd,
        parse_outfile_method=Parser.KEY_VALUE,
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
        root_directory=tmp_wd,
        command_line=exec_cmd,
        parse_outfile_method=Parser.KEY_VALUE,
        input_filename="input.cfg",
        output_filename="output.cfg",
        directory_naming_method=DirectoryNamingMethod.UUID,
    )

    disc.execute(indata)


@pytest.mark.parametrize(
    "parser",
    [
        Parser.TEMPLATE,
        Parser.KEY_VALUE,
        zip,
    ],
)
def test_disc_from_exe_cfgobj_parser_str(tmp_wd, parser) -> None:
    """Test the instantiation of the discipline with built-in and custom parsers."""
    sum_path = join(DIRNAME, "cfgobj_exe.py")
    exec_cmd = f"python {sum_path} -i input.cfg -o output.cfg"

    create_discipline(
        "DiscFromExe",
        input_template=join(DIRNAME, "input_template.cfg"),
        output_template=join(DIRNAME, "output_template.cfg"),
        root_directory=tmp_wd,
        command_line=exec_cmd,
        parse_outfile_method=parser,
        input_filename="input.cfg",
        output_filename="output.cfg",
        directory_naming_method="UUID",
    )


def test_exec_cfg(tmp_wd) -> None:
    outfile = "out_dummy.cfg"
    infile = join(DIRNAME, "input.cfg")
    exec_cfg(infile, outfile)


def test_exec_json(tmp_wd) -> None:
    outfile = "out_dummy.json"
    infile = join(DIRNAME, "input.json")
    exec_sum(infile, outfile)


def test_disc_from_exe_fail_exe(tmp_wd) -> None:
    sum_path = join(DIRNAME, "cfgobj_exe_fails.py")
    exec_cmd = "python " + sum_path + " -i input.cfg -o output.cfg -f wrong_len"
    disc: DiscFromExe = create_discipline(
        "DiscFromExe",
        input_template=join(DIRNAME, "input_template.cfg"),
        output_template=join(DIRNAME, "output_template.cfg"),
        root_directory=tmp_wd,
        command_line=exec_cmd,
        parse_outfile_method=Parser.KEY_VALUE,
        input_filename="input.cfg",
        output_filename="output.cfg",
    )

    indata = {"input 1": array([1]), "input 2": array([3]), "input 3": array([2])}
    with pytest.raises(ValueError):
        disc.execute(indata)

    disc.execution_status.value = disc.execution_status.Status.DONE
    exec_cmd = "python " + sum_path + " -i input.cfg -o output.cfg -f err_code"
    disc._executable_runner.command_line = exec_cmd
    with pytest.raises(CalledProcessError):
        disc.execute(indata)


def test_parse_key_value_file() -> None:
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


def test_parse_outfile() -> None:
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


def test_command_line(tmp_wd) -> None:
    """Test the property: ``command_line``."""
    sum_path = join(DIRNAME, "cfgobj_exe_fails.py")
    exec_cmd = "python " + sum_path + " -i input.cfg -o output.cfg -f wrong_len"
    disc: DiscFromExe = create_discipline(
        "DiscFromExe",
        input_template=join(DIRNAME, "input_template.cfg"),
        output_template=join(DIRNAME, "output_template.cfg"),
        root_directory=tmp_wd,
        command_line=exec_cmd,
        parse_outfile_method=Parser.KEY_VALUE,
        input_filename="input.cfg",
        output_filename="output.cfg",
    )
    assert disc.command_line == exec_cmd


def test_parallel_execution(tmp_wd) -> None:
    """Check if a :class:`~.DiscFromExe` executed within a multiprocess DOE can generate
    unique folders in :attr:`~.DirectoryNamingMethod.NUMBERED` mode.

    The check is focused on this topic since the multiprocess features of
    :class:`~.DiscFromExe` are used there.
    """
    nb_process = 2
    sum_path = join(DIRNAME, "sum_data.py")
    exec_cmd = f"python {sum_path} -i input.json -o output.json"

    disc = create_discipline(
        "DiscFromExe",
        input_template=join(DIRNAME, "input.json.template"),
        output_template=join(DIRNAME, "output.json.template"),
        root_directory=tmp_wd,
        command_line=exec_cmd,
        input_filename="input.json",
        output_filename="output.json",
        directory_naming_method=DirectoryNamingMethod.UUID,
    )

    design_space = DesignSpace()
    design_space.add_variable("a", 1, lower_bound=1, upper_bound=2, value=1.02)

    scenario = create_scenario(
        disc,
        "out",
        design_space,
        scenario_type="DOE",
        formulation_name="DisciplinaryOpt",
    )

    scenario.execute(algo_name="OT_LHS", n_samples=2, n_processes=nb_process)

    assert len(tuple(tmp_wd.iterdir())) == nb_process


@pytest.mark.parametrize("clean_after_execution", [True, False])
def test_working_directory(tmp_wd, clean_after_execution: bool) -> None:
    """Test the property: ``working_directory``."""
    sum_path = join(DIRNAME, "cfgobj_exe.py")
    exec_cmd = f"python {sum_path} -i input.cfg -o output.cfg"

    disc: DiscFromExe = create_discipline(
        "DiscFromExe",
        input_template=join(DIRNAME, "input_template.cfg"),
        output_template=join(DIRNAME, "output_template.cfg"),
        root_directory=tmp_wd,
        command_line=exec_cmd,
        parse_outfile_method=Parser.KEY_VALUE,
        input_filename="input.cfg",
        output_filename="output.cfg",
        clean_after_execution=clean_after_execution,
    )

    assert disc._executable_runner.working_directory is None
    disc.execute({
        "input 1": array([1.0]),
        "input 2": array([3.0]),
        "input 3": array([2.0]),
    })
    assert disc._executable_runner.working_directory == tmp_wd / "1"

    if clean_after_execution:
        assert not disc._executable_runner.working_directory.is_dir()
    else:
        assert disc._executable_runner.working_directory.is_dir()
