# Contributor Instructions

*You must have a C compiler installed on your system.*

1. Clone the repository using `git clone https://github.com/MattTheCuber/edge-mender.git`
2. Initialize the submodule using `git submodule update --init`
3. [Install uv](https://docs.astral.sh/uv/getting-started/installation)
4. Create a virtual environment using `uv venv`
5. Install all development dependencies using `uv sync --all-extras`
6. Compile the Cython code using `python setup.py build_ext --inplace`
7. Run `pre-commit install`
8. Create a branch and start writing code!
