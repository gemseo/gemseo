set dotenv-load
set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# Most recipes use the Python version defined by .python-version,
# override it with UV_PYTHON (or .env).

# The oldest supported Python version, used to test the lowest dependency bounds,
# it shall mirror .python-version.
min-python := '3.10'

# Recursive invocations shall use the very same just binary and justfile.
just := quote(just_executable()) + ' --justfile ' + quote(justfile())

[private]
default:
    @{{ just }} --list

# Run the tests
[group('qa')]
test *args:
    uv run --extra all pytest {{ args }}

# Run the tests with coverage
[group('qa')]
coverage *args:
    @{{ just }} test --cov --cov-report=xml --cov-report=html --no-cov-on-fail {{ args }}

# Run `diff-cover` to check the coverage of the modified lines, e.g. `just diff-cover upstream/develop`
[group('qa')]
diff-cover compare-branch *args:
    uv run diff-cover coverage.xml --compare-branch {{ compare-branch }} --branch-coverage --fail-under=100 --format=html:diff-cover.html {{ args }}

# Run the coverage and then `diff-cover`, e.g. `just coverage-diff upstream/develop`
[group('qa')]
coverage-diff compare-branch *args:
    @{{ just }} coverage {{ args }}
    @{{ just }} diff-cover {{ compare-branch }}

# numpy and scipy are direct dependencies but also transitive ones,
# for which uv uses the highest versions;
# bounding them below makes uv use their lowest versions,
# these bounds shall mirror the ones of pyproject.toml.
[doc('Run the tests against the minimum dependency versions')]
[group('qa')]
test-min-deps *args:
    uv run --python {{ min-python }} --isolated --no-dev --group test --extra all --resolution lowest-direct --with "numpy>=1.24,<2" --with "scipy>=1.15,<1.16" pytest {{ args }}

# Install the git hooks
[group('qa')]
install-hooks:
    uv run --only-group check prek install

# Run the code formatting and checking
[group('qa')]
check *args: install-hooks
    uv run --only-group check prek run --all-files {{ args }}

# Run the static type checker
[group('qa')]
check-typing *args:
    uv run --only-group typing mypy {{ args }}

# Build and serve the documentation
[env("DISABLE_MKDOCS_2_WARNING", "true")]
[env("DOCSTRING_INHERITANCE_ENABLE", "1")]
[group('doc')]
doc *args:
    uv run --group doc --extra all mkdocs serve {{ args }}

# Build and serve the documentation without the API
[group('doc')]
doc-fast *args:
    @{{ just }} doc --config-file mkdocs-fast.yml {{ args }}

# Create and check the PyPI distribution
[group('packaging')]
dist:
    uv build --clear
    uv run --only-group dist check-wheel-contents dist --ignore W002

# Upload the distribution to the package repository
[group('packaging')]
publish: dist
    uv publish

# Update the dependencies and the git hooks
[group('lifecycle')]
update:
    uv lock --upgrade
    uv run --only-group check prek autoupdate

# Ensure the project virtualenv is up to date and has the base dependencies
[group('lifecycle')]
install *args:
    uv sync --extra all {{ args }}

# Remove the temporary files
[group('lifecycle')]
[unix]
clean:
    rm -rf .venv .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov diff-cover.html dist site
    find . -type d -name __pycache__ -prune -exec rm -rf {} +

# Remove the temporary files
[group('lifecycle')]
[windows]
clean:
    Remove-Item -Recurse -Force -ErrorAction Ignore .venv, .pytest_cache, .mypy_cache, .ruff_cache, .coverage, coverage.xml, htmlcov, diff-cover.html, dist, site
    Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force -ErrorAction Ignore

# Recreate the project virtualenv from scratch
[group('lifecycle')]
fresh: clean install
