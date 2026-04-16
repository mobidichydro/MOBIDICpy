# `mobidicpy` developer documentation

## Development install

```shell
# Create a virtual environment, e.g. with
python -m venv env

# activate virtual environment
source env/bin/activate

# make sure to have a recent version of pip and setuptools
python -m pip install --upgrade pip setuptools

# (from the project root directory)
# install mobidicpy as an editable package with development dependencies
python -m pip install --no-cache-dir --editable .[dev]
```

Alternatively, if you prefer using the Makefile:

```shell
make install-dev
```

Afterwards check that the install directory is present in the `PATH` environment variable.

## Running the tests

There are two ways to run tests.

The first way requires an activated virtual environment with the development tools installed:

```shell
pytest -v
```

Alternatively, using the Makefile:

```shell
make test
```


### Test coverage

In addition to just running the tests to see if they pass, they can be used for coverage statistics, i.e. to determine how much of the package's code is actually executed during tests.
In an activated virtual environment with the development tools installed, inside the package directory, run:

```shell
coverage run
coverage report
```

Or using the Makefile:

```shell
make coverage
```

To generate an HTML coverage report:

```shell
make coverage-html
```

`coverage` can also generate output in HTML and other formats; see `coverage help` for more information.

## Running linters locally

For linting and code formatting we use [ruff](https://docs.astral.sh/ruff/). Running the linters requires an
activated virtual environment with the development tools installed.

```shell
# Check code quality
ruff check .

# Fix code quality issues automatically
ruff check . --fix

# Format code
ruff format .
```

Or using the Makefile:

```shell
# Check code quality
make lint-check

# Fix code quality issues automatically
make lint

# Format code
make format
```

## Testing docs locally

To build the documentation locally, first make sure `mkdocs` and its dependencies are installed:
```shell
python -m pip install .[doc]
```

Or using the Makefile:
```shell
make install-doc
```

Then you can build the documentation and serve it locally with:
```shell
mkdocs serve
```

Or using the Makefile:
```shell
make docs-serve
```

This will return a URL (e.g. `http://127.0.0.1:8000/mobidicpy/`) where the docs site can be viewed.

## Versioning

Bumping the version across all files is done with [bump-my-version](https://github.com/callowayproject/bump-my-version), e.g.

```shell
bump-my-version bump major  # bumps from e.g. 0.3.2 to 1.0.0
bump-my-version bump minor  # bumps from e.g. 0.3.2 to 0.4.0
bump-my-version bump patch  # bumps from e.g. 0.3.2 to 0.3.3
```

Or using the Makefile:

```shell
make bump-major  # bumps from e.g. 0.3.2 to 1.0.0
make bump-minor  # bumps from e.g. 0.3.2 to 0.4.0
make bump-patch  # bumps from e.g. 0.3.2 to 0.3.3
```

After bumping the version, push the commit and tag to GitHub:

```shell
# --follow-tags pushes only annotated tags reachable from the commits being pushed
git push origin main --follow-tags
```

## Making a release

This section describes how to make a release in 2 parts:

1. Preparation
2. Making a release on GitHub

### (1/2) Preparation

1. Verify that the information in CITATION.cff is correct.
2. Make sure the [version has been updated](#versioning).
3. Run the unit tests with `pytest -v`


### (2/2) GitHub

Make a [release on GitHub](https://github.com/mobidichydro/mobidicpy/releases/new). 
