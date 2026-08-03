set dotenv-load
set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

python := env('UV_PYTHON', '3.10')

# Set environment variables
# TODO: only in recipes that matters
export WINDIR := env('WINDIR', 'C:\Windows')
export GEMSEO_KEEP_IMAGE_COMPARISONS := env('GEMSEO_KEEP_IMAGE_COMPARISONS', '')

@_:
    just --list

# Run tests
[group('qa')]
test *args:
    uv run --python {{python}} --extra all pytest {{args}}

# Run tests coverage
[group('qa')]
coverage *args:
    just test --cov --cov-report=xml --cov-report=html --no-cov-on-fail {{args}}

# Test with minimum dependency versions
[group('qa')]
test-min-deps *args:
    # UV_PROJECT_ENVIRONMENT=.venv-min-deps uv run --no-dev --group test --extra all --resolution lowest-direct --python 3.10 pytest {{args}}
    # numpy and scipy are direct dependencies but also transitive ones,
    # for which uv uses the highest versions;
    # bounding them here makes uv use their lowest versions,
    # these bounds shall mirror the ones of pyproject.toml.
    uv run --python 3.10 --isolated --no-dev --group test --extra all --resolution lowest-direct --with "numpy>=1.24,<2" --with "scipy>=1.15,<1.16" pytest {{args}}

# Run code formatting and checking
[group('qa')]
check *args:
    uv run --only-group check prek install
    uv run --only-group check prek run --all-files {{args}}

# Run code formatting and checking
[group('qa')]
check-typing *args:
    uv run --only-group typing mypy {{args}}

# Build and serve documentation
[group('doc'),
env("DOCSTRING_INHERITANCE_ENABLE", "1"),
env("DISABLE_MKDOCS_2_WARNING", "true")]
doc *args:
    uv run --python {{python}} --group doc --extra all mkdocs serve {{args}}

# Build and serve documentation without API
[group('doc')]
doc-fast *args:
    just doc --config-file mkdocs-fast.yml {{args}}

# Create and check PyPI distribution
[group('packaging')]
dist:
    uv build --clear
    uv run --only-group dist check-wheel-contents dist --ignore W002

# Upload distribution to package repository
[group('packaging')]
publish: dist
    uv publish

# Update dependencies
[group('lifecycle')]
update:
    uv lock --upgrade
    uv run --only-group check prek autoupdate

# Ensure project virtualenv is up to date and has the base dependencies
[group('lifecycle')]
install *args:
    uv sync --python {{python}} --group check --extra all {{args}}

# Remove temporary files
[group('lifecycle')]
clean:
    rm -rf .venv .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
    find . -type d -name "__pycache__" -exec rm -r {} +

# Recreate project virtualenv from scratch
[group('lifecycle')]
fresh: clean install
