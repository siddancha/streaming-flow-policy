# Streaming Flow Policy

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

### Via PDM (recommended for development)

2. Install PDM.
    ```bash
    pip install pdm
    ```

3. Run pdm-install:
    ```bash
    pdm install
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

#### To deploy to MIT website

```bash
scp -r docs/_build/html/* $WEBSITE_HOME/notebooks/
```
where `$WEBSITE_HOME` is `sancha@athena.dialup.mit.edu:/afs/athena.mit.edu/user/s/a/sancha/www`.
