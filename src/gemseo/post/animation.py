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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Simone Coniglio
"""Make an animated GIF from a :class:`.BasePost`."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import ClassVar
from typing import Final

from PIL import Image

from gemseo.algos.database import Database
from gemseo.post.animation_settings import Animation_Settings
from gemseo.post.base_post import BasePost


class Animation(BasePost[Animation_Settings]):
    """Animated GIF maker from a :class:`.BasePost`."""

    __FRAME: Final[str] = "frame"
    """The prefix for frame images."""

    Settings: ClassVar[type[Animation_Settings]] = Animation_Settings

    def _plot(
        self,
        settings: Animation_Settings,
    ) -> None:
        steps_to_frame_file_paths = self.__generate_frames(settings)
        output_files = self.__generate_gif(
            steps_to_frame_file_paths,
            settings,
        )
        self._output_files = output_files

        if settings.remove_frames:
            for file_paths in steps_to_frame_file_paths:
                for file_path in file_paths:
                    Path(file_path).unlink()

        self._output_files += steps_to_frame_file_paths

    def __generate_frames(
        self,
        settings: Animation_Settings,
    ) -> list[list[Path]]:
        """Generate the frames of the animation.

        Args:
            settings: The post-processing settings.

        Returns:
            The paths to the images at each time step of the animation.
        """
        steps_to_frame_file_paths = []
        opt_problem = self.optimization_problem

        temporary_database = settings.temporary_database_path
        if temporary_database:
            temporary_database = Path(temporary_database)
            database = opt_problem.database
            database.to_hdf(temporary_database)
        else:
            database = deepcopy(opt_problem.database)

        output_files_count = 0
        for iteration in range(len(database), 0, -settings.frame_rate):
            opt_problem.database.clear_from_iteration(iteration)
            settings.post_processing_settings.file_path = f"{self.__FRAME}_{iteration}"
            settings.post_processing.execute(
                **settings.post_processing_settings.model_dump(),
            )
            steps_to_frame_file_paths.append(
                settings.post_processing.output_file_paths[output_files_count:],
            )
            output_files_count = len(settings.post_processing.output_file_paths)

        if temporary_database:
            opt_problem.database = Database().from_hdf(temporary_database)
            temporary_database.unlink()
        else:
            opt_problem.database = database

        return steps_to_frame_file_paths[::-1]

    def __generate_gif(
        self,
        steps_to_frame_file_paths: list[list[Path]],
        settings: Animation_Settings,
    ) -> list[str]:
        """Generate and store the GIF using input frames.

        Args:
            steps_to_frame_file_paths: The frame file paths for the different time
                steps.
            settings: The post-processing settings.

        Returns:
            The output file paths.
        """
        gif_file_path = Path(settings.gif_file_path)
        figure_names_to_frames = {}
        for step, frame_file_paths in enumerate(steps_to_frame_file_paths):
            if len(frame_file_paths) > 1:
                figure_names = [
                    Path(frame_file_path).stem.replace(
                        f"{self.__FRAME}_{step + 1}_", ""
                    )
                    for frame_file_path in frame_file_paths
                ]
            else:
                figure_names = [self.__FRAME]
            frames = [
                Image.open(fp=frame_file_path) for frame_file_path in frame_file_paths
            ]

            for figure_name, frame in zip(figure_names, frames):
                if figure_name not in figure_names_to_frames:
                    figure_names_to_frames[figure_name] = []
                figure_names_to_frames[figure_name].append(frame)

        if len(figure_names_to_frames) > 1:
            file_paths_to_frames = {
                f"{gif_file_path.stem}_{suffix}.gif": figure_names_to_frames[suffix]
                for suffix in figure_names
            }
        else:
            file_paths_to_frames = {
                f"{gif_file_path.stem}.gif": next(iter(figure_names_to_frames.values()))
            }

        output_file_paths = []
        first_iteration = settings.first_iteration

        for file_path, frames in file_paths_to_frames.items():
            if first_iteration > 0:
                first_iteration -= 1
            frames = frames[first_iteration:] + frames[:first_iteration]
            frames[0].save(
                file_path,
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=settings.time_step,
                loop=settings.n_repetitions,
            )
            output_file_paths.append(file_path)

        return output_file_paths
