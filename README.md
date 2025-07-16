<div align="center">

<div id="user-content-toc" style="margin-bottom: 10px">
      <h1 >Streaming Flow Policy</h1>
      <h3>Simplifying diffusion/flow-matching policies by treating action trajectories as flow trajectories
      <h4><a href="https://streaming-flow-policy.github.io/">🌐 Website</a>  &nbsp;•&nbsp;  <a href=https://arxiv.org/abs/2505.21851>📄 Paper</a> &nbsp;•&nbsp; <a href="https://youtu.be/gqUnEzBCbZE">🎬 Talk</a> &nbsp;•&nbsp; <a href=https://x.com/siddancha/status/1925170490856833180>🐦 Twitter</a> &nbsp;•&nbsp; <a href=https://siddancha.github.io/streaming-flow-policy/notebooks>📚 Notebooks</a></h4>
</h3>
<img width=80% src="https://github.com/user-attachments/assets/48e88da0-97fa-4a99-9ecf-54258fa45c0f"></img>
</div>
</div>

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
