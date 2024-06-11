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
"""Job schedulers interface."""

from __future__ import annotations

import pickle
from logging import getLogger
from pathlib import Path
from string import Template
from subprocess import CompletedProcess
from subprocess import run as subprocess_run
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from uuid import uuid1

from gemseo.core.discipline import MDODiscipline

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.typing import JacobianData
    from gemseo.typing import StrKeyMapping

LOGGER = getLogger(__name__)


class JobSchedulerDisciplineWrapper(MDODiscipline):
    """A discipline that wraps the execution with job schedulers.

    The discipline is serialized to the disk, its inputs too, then a job file is created
    from a template to execute it with the provided options. The submission command is
    launched, it will set up the environment, deserialize the discipline and its inputs,
    execute it and serialize the outputs. Finally, the deserialized outputs are returned
    by the wrapper.
    """

    DISC_PICKLE_FILE_NAME: ClassVar[str] = "discipline.pckl"
    DISC_INPUT_FILE_NAME: ClassVar[str] = "input_data.pckl"
    DISC_OUTPUT_FILE_NAME: ClassVar[str] = "output_data.pckl"
    TEMPLATES_DIR_PATH: ClassVar[Path] = Path(__file__).parent / "templates"

    _discipline: MDODiscipline
    """The discipline to wrap in the job scheduler."""
    _job_template_path: Path
    """The path to the template to be used to make a submission to the job scheduler
    command."""
    _workdir_path: Path
    """The path to the workdir where the files will be generated."""
    _scheduler_run_command: str
    """The command to call the job scheduler and submit the generated script."""
    _job_out_filename: str
    """The output job file name."""
    _options: dict[str, Any]
    """The job scheduler specific options."""
    _setup_cmd: str
    """The environment command to be used before running."""
    _execute_at_linearize: bool
    """Whether to execute the discipline when linearizing."""

    def __init__(
        self,
        discipline: MDODiscipline,
        workdir_path: Path,
        scheduler_run_command: str = "sbatch --wait",
        job_out_filename: str = "batch.srun",
        job_template_path: Path | str | None = None,
        use_template: bool = True,
        setup_cmd: str = "",
        **options,
    ) -> None:
        """
        Args:
            discipline: The discipline to wrap in the job scheduler.
            workdir_path: The path to the workdir where the files will be generated.
            scheduler_run_command: The command to call the job scheduler and submit
                the generated script.
            job_out_filename: The output job file name.
            job_template_path: The path to the template to be used to make a
                submission to the job scheduler command.
            use_template: whether to use template based interface to the job scheduler.
            setup_cmd: The command used before running the executable.
            **options: The job scheduler specific options to be used in the template.

        Raises:
            OSError: If ``job_template_path`` does not exist.
        """  # noqa:D205 D212 D415
        super().__init__(discipline.name, grammar_type=discipline.grammar_type)
        self._discipline = discipline
        self._use_template = use_template
        self._job_template_path = job_template_path
        if job_template_path is not None and not isinstance(job_template_path, Path):
            self._job_template_path = Path(self._job_template_path)

        self._scheduler_run_command = scheduler_run_command
        self._job_out_filename = job_out_filename
        self._setup_cmd = setup_cmd
        self._options = options

        self.input_grammar = self._discipline.input_grammar
        self.output_grammar = self._discipline.output_grammar
        self.default_inputs = self._discipline.default_inputs
        self._workdir_path = workdir_path
        self.pickled_discipline = pickle.dumps(self._discipline)
        self.job_file_template = None
        self._execute_at_linearize = True

        if use_template:
            self._parse_template()

    def _parse_template(self) -> None:
        """Parse the template.

        Either it is passed directly by the user, or, if None, tries to find it in
        the templates' directory. The file name must then be the class name.

        Raises:
            FileNotFoundError: If ``job_template_path`` does not exist.
        """
        if self._job_template_path is None:
            self._job_template_path = self.TEMPLATES_DIR_PATH / self.__class__.__name__
        if not self._job_template_path.exists():
            msg = (
                f"Job scheduler template file {self._job_template_path} does not exist."
            )
            raise FileNotFoundError(msg)
        self.job_file_template = Template(self._job_template_path.read_text())

    def _generate_job_file_from_template(
        self,
        current_workdir: Path,
        discipline_path: Path,
        inputs_path: Path,
        outputs_path: Path,
        log_path: Path,
        linearize: bool,
    ) -> Path:
        """Generate the job file from the template.

        Args:
            current_workdir: The path to the workdir where the files will be generated.
            discipline_path: The path to the serialized discipline.
            inputs_path: The path to the serialized input data.
            outputs_path: The path to the serialized output data.
            log_path: The path to the log file generated by the job scheduler.
            linearize: Whether to linearize the discipline.

        Returns:
            The destination job file path.
        """
        linearize_option = "--linearize" if linearize else ""

        if linearize and self._execute_at_linearize:
            execute_at_linearize_option = "--execute-at-linearize"
        else:
            execute_at_linearize_option = ""

        try:
            job_file_content = self.job_file_template.substitute(
                discipline_name=self._discipline.name,
                log_path=log_path,
                setup_cmd=self._setup_cmd,
                workdir_path=str(current_workdir),
                discipline_path=str(discipline_path),
                inputs_path=str(inputs_path),
                outputs_path=str(outputs_path),
                linearize=linearize_option,
                execute_at_linearize=execute_at_linearize_option,
                **self._options,
            )
        except KeyError as err:
            msg = f"Value not passed to template for key: {err}"
            raise KeyError(msg) from err
        dest_job_file_path = current_workdir / self._job_out_filename
        dest_job_file_path.write_text(job_file_content, encoding="utf8")
        return dest_job_file_path

    def _create_run_command(
        self,
        current_workdir: Path,
        dest_job_file_path: Path | str,
    ) -> str:
        """Create the scheduler submission command.

        Args:
            current_workdir: The current working directory.
            dest_job_file_path: The destination job scheduler input file path.

        Returns:
            The command.
        """
        return f"{self._scheduler_run_command} {dest_job_file_path}"

    def _run_command(
        self,
        current_workdir: Path,
        dest_job_file_path: Path | str,
    ) -> CompletedProcess:
        """Run the scheduler submission command.

        Args:
            current_workdir: The current working directory.
            dest_job_file_path: The destination job scheduler input file path.

        Returns:
            The return code of the command.

        Raises:
            CalledProcessError: When the command failed.
        """
        cmd = self._create_run_command(current_workdir, dest_job_file_path)
        LOGGER.debug("Submitting the job command: %s", cmd)
        completed = subprocess_run(
            cmd.split(),
            capture_output=True,
            cwd=current_workdir,
        )

        if completed.returncode != 0:
            LOGGER.error(
                "Failed to submit the job command %s, for discipline %s "
                "in the working directory %s",
                cmd,
                self._discipline.name,
                current_workdir,
            )

        completed.check_returncode()

        LOGGER.debug("Job execution ended in %s", current_workdir)
        return completed

    def _handle_outputs(
        self,
        current_workdir: Path,
        outputs_path: Path,
    ) -> None:
        """Read the serialized outputs.

        If an exception is contained inside, raises it.

        If the outputs contain data, it updates self.local_data with it.

        Args:
            current_workdir: The current working directory.
            outputs_path: The path to the serialized output data.

        Raises:
            FileNotFoundError: When the outputs contain an error.
        """
        if not outputs_path.exists():
            msg = (
                "The serialized outputs file of the discipline does not exist: "
                f"{outputs_path}."
            )
            raise FileNotFoundError(msg)

        with outputs_path.open("rb") as output_file:
            output = pickle.load(output_file)

            if isinstance(output[0], BaseException):
                error, trace = output
                LOGGER.error(
                    "Discipline %s execution failed in %s",
                    self._discipline.name,
                    current_workdir,
                )

                LOGGER.error(trace)
                raise error

            LOGGER.debug(
                "Discipline %s execution succeeded in %s",
                self._discipline.name,
                current_workdir,
            )

            self.local_data.update(output[0])
            if output[1]:
                self.jac = output[1]

    def _create_current_workdir(self) -> Path:
        """Create the current working directory.

        Returns:
            The path to the working directory.
        """
        # This generates a unique random and thread safe working directory name.
        # We do not use tempdir from the standard python library because it is a
        # permanent run directory.
        loc_id = str(uuid1()).split("-")[0]
        current_workdir = Path(self._workdir_path / loc_id)
        current_workdir.mkdir()
        return current_workdir

    def _write_inputs_to_disk(
        self,
        current_workdir: Path,
        differentiated_inputs: Iterable[str],
        differentiated_outputs: Iterable[str],
    ) -> tuple[Path, Path]:
        """Write the serialized input data to the current working directory.

        Args:
            current_workdir: The current working directory.
            differentiated_inputs: If the linearization is performed, the
                inputs that define the rows of the jacobian.
            differentiated_outputs: If the linearization is performed, the
                outputs that define the columns of the jacobian.

        Returns:
            The path to the serialized discipline and inputs.
        """
        discipline_path = current_workdir / self.DISC_PICKLE_FILE_NAME
        discipline_path.write_bytes(self.pickled_discipline)
        inputs_path = current_workdir / self.DISC_INPUT_FILE_NAME
        serialized_data = pickle.dumps((
            self.local_data,
            differentiated_inputs,
            differentiated_outputs,
        ))
        inputs_path.write_bytes(serialized_data)
        return discipline_path, inputs_path

    def _wait_job(self, current_workdir: Path) -> None:
        """Wait for the end of the job.

        By default, does nothing and expect the run command to be blocking.

        Args:
            current_workdir: The path to the workdir where the files will be generated.
        """

    def _run_or_compute_jacobian(
        self,
        linearize: bool,
        differentiated_inputs: Iterable[str] = (),
        differentiated_outputs: Iterable[str] = (),
    ) -> None:
        """Executes or linearizes the discipline.

        Args:
            linearize: Whether to linearize the discipline.
            differentiated_inputs: If the linearization is performed, the
                inputs that define the rows of the jacobian.
            differentiated_outputs: If the linearization is performed, the
                outputs that define the columns of the jacobian.
        """
        current_workdir = self._create_current_workdir()
        outputs_path = current_workdir / self.DISC_OUTPUT_FILE_NAME
        log_path = current_workdir / f"{self._discipline.name}.log"
        discipline_path, inputs_path = self._write_inputs_to_disk(
            current_workdir, differentiated_inputs, differentiated_outputs
        )

        if self._use_template:
            dest_job_file_path = self._generate_job_file_from_template(
                current_workdir,
                discipline_path,
                inputs_path,
                outputs_path,
                log_path,
                linearize,
            )
        else:
            dest_job_file_path = ""

        self._run_command(current_workdir, dest_job_file_path)
        self._wait_job(current_workdir)
        self._handle_outputs(current_workdir, outputs_path)

    def _run(self) -> None:
        self._run_or_compute_jacobian(False)

    def linearize(  # noqa: D102
        self,
        input_data: StrKeyMapping | None = None,
        compute_all_jacobians: bool = False,
        execute: bool = True,
    ) -> JacobianData:
        self._execute_at_linearize = execute
        return super().linearize(
            input_data=input_data,
            compute_all_jacobians=compute_all_jacobians,
            execute=False,
        )

    def _compute_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        self._run_or_compute_jacobian(
            True, differentiated_inputs=inputs, differentiated_outputs=outputs
        )
