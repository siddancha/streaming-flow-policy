# Flow Policy

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

#### To deploy to MIT website

```bash
scp -r docs/_build/html/* $WEBSITE_HOME/notebooks/
```
where `$WEBSITE_HOME` is `sancha@athena.dialup.mit.edu:/afs/athena.mit.edu/user/s/a/sancha/www`.
