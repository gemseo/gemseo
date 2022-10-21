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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author:  Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pickle
from pathlib import Path

import pytest
from gemseo.api import create_discipline
from gemseo.core.discipline import MDODiscipline
from gemseo.post.core.gantt_chart import create_gantt_chart
from gemseo.utils.testing import image_comparison

TIME_STAMPS_PATH = Path(__file__).parent / "time_stamps.pickle"


@pytest.fixture
def reset_time_stamping():
    """Reset the time stamping before and after a test."""
    MDODiscipline.deactivate_time_stamps()
    MDODiscipline.activate_time_stamps()
    yield
    MDODiscipline.deactivate_time_stamps()


@pytest.fixture(scope="module")
def time_stamps_data():
    """Return the reference time stamps from local pickle."""
    with TIME_STAMPS_PATH.open("rb") as infile:
        data = pickle.load(infile)
    return data


def test_time_stamps(reset_time_stamping):
    """Tests the time stamps storage."""
    mission = create_discipline("SobieskiMission", enable_delay=True)
    mission.execute()
    data = {"x_shared": mission.default_inputs["x_shared"] + 1.0}
    mission.linearize(data, force_all=True)
    stamps = MDODiscipline.time_stamps

    assert "SobieskiMission" in stamps
    assert len(stamps) == 1

    mission_stamps = stamps["SobieskiMission"]
    assert len(mission_stamps) == 3

    for stamp in mission_stamps:
        if not stamp[2]:
            assert stamp[1] - stamp[0] > 0.9

    assert not mission_stamps[0][-1]
    assert not mission_stamps[1][-1]
    assert mission_stamps[2][-1]


def test_stamps_error():
    """Tests that the error is raised when time stamps are deactivated."""
    with pytest.raises(
        ValueError, match="Time stamps are not activated in MDODiscipline"
    ):
        create_gantt_chart()


def test_save(tmp_wd, pyplot_close_all, reset_time_stamping, time_stamps_data):
    """Tests file saving."""
    MDODiscipline.time_stamps = time_stamps_data
    file_path = Path("gantt_chart.png")
    create_gantt_chart(file_path=file_path, font_size=10)
    assert file_path.exists()


@image_comparison(["gantt_chart"])
def test_plot(tmp_wd, pyplot_close_all, reset_time_stamping, time_stamps_data):
    """Tests the Gantt chart plot creation."""
    # If needed for figure regeneration:
    #
    # disciplines = create_discipline(
    # [
    # "SobieskiPropulsion",
    # "SobieskiAerodynamics",
    # "SobieskiMission",
    # "SobieskiStructure",
    # ]
    # )
    #
    # design_space = SobieskiProblem().design_space
    # scenario = create_scenario(
    # disciplines,
    # "MDF",
    # objective_name="y_4",
    # design_space=design_space,
    # maximize_objective=True,
    # )
    # for c_name in ["g_1", "g_2", "g_3"]:
    # scenario.add_constraint(c_name, "ineq")
    # scenario.execute({"max_iter": 3, "algo": "SLSQP"})
    #
    # stamps = MDODiscipline.TIME_STAMPS
    # with open(TIME_STAMPS_PATH, "wb") as outfile:
    # pickle.dump(stamps, outfile)
    MDODiscipline.time_stamps = time_stamps_data
    create_gantt_chart(save=False, font_size=10)


@image_comparison(["gantt_chart_filtered"])
def test_plot_filter(tmp_wd, pyplot_close_all, reset_time_stamping, time_stamps_data):
    """Tests the Gantt chart plot creation with disciplines filter."""
    MDODiscipline.time_stamps = time_stamps_data
    create_gantt_chart(
        save=False,
        font_size=10,
        disc_names=["SobieskiPropulsion", "SobieskiAerodynamics"],
    )


def test_plot_filter_fail(
    tmp_wd, pyplot_close_all, reset_time_stamping, time_stamps_data
):
    """Tests the Gantt chart disciplines filter failure."""
    MDODiscipline.time_stamps = time_stamps_data
    with pytest.raises(ValueError, match="have no time stamps"):
        create_gantt_chart(save=False, disc_names=["IDONTEXIST"])
