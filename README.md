# Streaming Flow Policy
### Simplifying diffusion/flow-matching policies by treating action trajectories as flow trajectories

## Installation

1. Create a virtual environment
    ```bash
    python3 -m venv .venv --prompt=streaming-flow-policy
    source .venv/bin/activate
    ```

### Via pip

2. pip-install this repository.
    ```bash
    pip install -e .
    ```

### Via uv (recommended for development)

2. Install [uv](https://docs.astral.sh/uv/).
    ```bash
    pip install uv
    ```

3. Sync Python dependencies using uv:
    ```bash
    uv sync
    ```


## Building Jupyter Book

The Jupyter Book is built using [jupyter-book](https://jupyterbook.org/intro.html). It lives in the `docs/` directory.

#### Command to clean the build directory.
```bash
jupyter-book clean docs
```

#### Command to build the book.
```bash
jupyter-book build docs
```

#### To add a notebook to the Jupyter book

Add a symlink to the `docs` directory.

#### View Jupyter book locally

The HTML content is created in the `docs/_build/html` directory.
