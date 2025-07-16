# Streaming Flow Policy
### Simplifying diffusion/flow-matching policies by treating action trajectories as flow trajectories

#### [🌐 Website](https://streaming-flow-policy.github.io/)  &nbsp;•&nbsp;  [📄 Paper](https://arxiv.org/abs/2505.21851) &nbsp;•&nbsp; [🎬 Talk](https://youtu.be/gqUnEzBCbZE) &nbsp;•&nbsp; [🐦 Twitter](https://x.com/siddancha/status/1925170490856833180) &nbsp;•&nbsp; [📚 Notebooks](https://siddancha.github.io/streaming-flow-policy/notebooks)

<img width=80% src="https://github.com/user-attachments/assets/48e88da0-97fa-4a99-9ecf-54258fa45c0f"></img>

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
