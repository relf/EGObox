# Getting Started

## Install the Python package

```bash
pip install egobox
```

## Use the Rust crates

```toml
[dependencies]
egobox-doe = { version = "0.x.y" }
egobox-gp = { version = "0.x.y" }
egobox-moe = { version = "0.x.y" }
egobox-ego = { version = "0.x.y" }
```

## Run the existing examples

```bash
cargo run -p egobox-ego --example xsinx --release
cargo run -p egobox-gp --example kriging --release
```

```bash
cd python
uv run python egobox/examples/website_egor.py
uv run python egobox/examples/website_gpx.py
```

## Work on the website locally

```bash
conda activate base
python --version
python -m pip install -r requirements-docs.txt
python -m mkdocs serve
```

The site reads its example snippets from repository files, so editing the example source updates the rendered page without copying code into Markdown.
Use a Conda environment with Python 3.12 for the website workflow.