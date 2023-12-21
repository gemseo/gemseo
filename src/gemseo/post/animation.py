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
"""Make an animated GIF from an :class:`.OptPostProcessor`."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
from typing import ClassVar
from typing import Final

from PIL import Image

from gemseo import execute_post
from gemseo.algos.database import Database
from gemseo.post.opt_post_processor import OptPostProcessor
from gemseo.post.opt_post_processor import OptPostProcessorOptionType
from gemseo.post.post_factory import PostFactory


class Animation(OptPostProcessor):
    """Animated GIF maker from an :class:`.OptPostProcessor`."""

    __OPT_POST_PROCESSOR: Final[str] = "opt_post_processor"
    """The name of opt_post_processor option."""
    _FRAME: Final[str] = "frame"
    """The  prefix for frame images."""
    __TEMPORARY_DATABASE_FILE: Final[str] = "temporary_database_file"
    """The name of temporary_database_file option."""
    DEFAULT_OPT_POST_PROCESSOR: ClassVar[str] = "BasicHistory"
    """The class name of the default :class:`.OptPostProcessor`."""

    def check_options(self, **options: OptPostProcessorOptionType) -> None:
        """Check the options to execute :class:`.Animation`.

        This includes the options of the :class:`.OptPostProcessor`.

        Args:
            **options: The options of the :class:`.Animation`.
        """
        self._update_grammar_from_class(
            cls=PostFactory().get_class(
                options.get(self.__OPT_POST_PROCESSOR, self.DEFAULT_OPT_POST_PROCESSOR)
            )
        )
        # cast paths as string or None before checks.
        options.update({
            self.__TEMPORARY_DATABASE_FILE: options.get(self.__TEMPORARY_DATABASE_FILE)
        })
        super().check_options(**options)

    def _plot(
        self,
        frame_rate: int = 1,
        opt_post_processor: str = DEFAULT_OPT_POST_PROCESSOR,
        first_iteration: int = -1,
        time_step: int = 100,
        n_repetitions: int | None = None,
        gif_file_path: str | Path = "animated_gif",
        temporary_database_file: str | Path | None = None,
        remove_frames: bool = True,
        **options: Any,
    ) -> list[str]:
        """Plot the frames, generate the animation, eventually remove frames.

        Args:
            frame_rate: The number of iterations per time step.
            opt_post_processor: The class name of the :class:`.OptPostProcessor`.
            first_iteration: The iteration to begin the animation.
            time_step: The time step between two frames in milliseconds.
            n_repetitions: The number of times the animation is played. If ``None``,
            infinitely.
            gif_file_path: The path to the GIF file.
            temporary_database_file:  The path to a temporary database file to avoid
                deepcopy memory errors. If ``None``, deepcopy is used instead.
            remove_frames: Wether to remove the frame images after the GIF generation.
            **options: The options of the :class:`.OptPostProcessor`.
        """
        steps_to_frame_file_paths = self._generate_frames(
            frame_rate, opt_post_processor, temporary_database_file, **options
        )
        output_files = self.__generate_gif(
            steps_to_frame_file_paths,
            gif_file_path,
            first_iteration,
            time_step,
            n_repetitions,
        )
        self._output_files = output_files
        if remove_frames:
            for file_paths in steps_to_frame_file_paths:
                for file_path in file_paths:
                    Path(file_path).unlink()
            return output_files

        self._output_files += steps_to_frame_file_paths
        return output_files + steps_to_frame_file_paths

    def _generate_frames(
        self,
        frame_rate: int,
        opt_post_processor: str,
        temporary_database_file: str | Path | None = None,
        **options: Any,
    ) -> list[list[str]]:
        """Generate the frames of the animation.

        Args:
            frame_rate: The rate of frame per iterations.
            opt_post_processor: The class name of the :class:`.OptPostProcessor`.
            temporary_database_file:  The path to a temporary database file to avoid
                deepcopy memory errors. If ``None`` or empty, deepcopy is used instead.
            **options: The options of the :class:`.OptPostProcessor`.

        Returns:
            The paths to the images at each time step of the animation.
        """
        steps_to_frame_file_paths = []
        opt_problem = self.opt_problem
        if temporary_database_file:
            temporary_database_file = Path(temporary_database_file)
            database = opt_problem.database
            database.to_hdf(temporary_database_file)
        else:
            database = deepcopy(opt_problem.database)

        for iteration in range(len(database), 0, -frame_rate):
            opt_problem.database.clear_from_iteration(iteration)
            options["file_path"] = f"{self._FRAME}_{iteration}"
            steps_to_frame_file_paths.append(
                execute_post(
                    opt_problem,
                    post_name=opt_post_processor,
                    **options,
                ).output_files,
            )

        if temporary_database_file:
            opt_problem.database = Database().from_hdf(temporary_database_file)
            temporary_database_file.unlink()
        else:
            opt_problem.database = database

        return steps_to_frame_file_paths[::-1]

    def __generate_gif(
        self,
        steps_to_frame_file_paths: list[list[str]],
        gif_file_path: str | Path,
        first_iteration: int,
        time_step: int,
        n_repetitions: int | None,
    ) -> list[str]:
        """Generate and store the GIF using input frames.

        Args:
            steps_to_frame_file_paths: The frame file paths for the different time
            steps.
            gif_file_path: The path to the GIF file.
            first_iteration: The iteration to begin the animation.
            time_step: The time step between two frames in milliseconds.
            n_repetitions: The number of times the animation is played. If ``None``,
            infinitely.

        Returns:
            The output file names.
        """
        gif_file_path = Path(gif_file_path)
        figure_names_to_frames = {}
        for step, frame_file_paths in enumerate(steps_to_frame_file_paths):
            if len(frame_file_paths) > 1:
                figure_names = [
                    Path(frame_file_path).stem.replace(f"{self._FRAME}_{step + 1}_", "")
                    for frame_file_path in frame_file_paths
                ]
            else:
                figure_names = [self._FRAME]
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
        for file_path, frames in file_paths_to_frames.items():
            if first_iteration > 0:
                first_iteration = first_iteration - 1
            frames = frames[first_iteration:] + frames[:first_iteration]
            frames[0].save(
                file_path,
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=time_step,
                loop=n_repetitions or 0,
            )
            output_file_paths.append(file_path)
        return output_file_paths
