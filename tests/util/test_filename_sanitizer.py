# Copyright 2007 Pallets
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1.  Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#
# 2.  Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#
# 3.  Neither the name of the copyright holder nor the names of its
#     contributors may be used to endorse or promote products derived from
#     this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This code is from
# https://werkzeug.palletsprojects.com/en/stable/utils/#werkzeug.utils.secure_filename
from __future__ import annotations

import pytest

from gemseo.util import _filename_sanitizer
from gemseo.util._filename_sanitizer import secure_filename
from gemseo.util.platform import PLATFORM_IS_WINDOWS


def test_secure_filename():
    """Test the secure_filename function."""
    assert secure_filename("My cool movie.mov") == "My_cool_movie.mov"
    assert secure_filename("../../../etc/passwd") == "etc_passwd"
    assert (
        secure_filename("i contain cool \xfcml\xe4uts.txt")
        == "i_contain_cool_umlauts.txt"
    )
    assert secure_filename("__filename__") == "filename"
    assert secure_filename("foo$&^*)bar") == "foobar"
    assert secure_filename("con.bar") == (
        "_con.bar" if PLATFORM_IS_WINDOWS else "con.bar"
    )


@pytest.fixture
def windows_os_name(monkeypatch) -> None:
    """Make the sanitizer behave as if it ran on Windows."""
    monkeypatch.setattr(_filename_sanitizer.os, "name", "nt")


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("con.bar", "_con.bar"),
        ("CON", "_CON"),
        ("lpt1.txt", "_lpt1.txt"),
        ("nul", "_nul"),
        ("not_a_device.txt", "not_a_device.txt"),
        ("...", ""),
    ],
)
@pytest.mark.usefixtures("windows_os_name")
def test_secure_filename_windows_device_files(filename: str, expected: str):
    """Verify that the names of the Windows device files are prefixed.

    Args:
        filename: The filename to sanitize.
        expected: The expected sanitized filename.
    """
    assert secure_filename(filename) == expected
