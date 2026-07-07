+++
title = "Getting Started"
+++

# Getting Started

## Python

### Install the package

```bash
pip install egobox
```
### Run the existing examples

```bash
cd python
uv run python egobox/examples/g24.py
uv run python egobox/examples/xsinx.py
```

## Rust

### Use the crates

```toml
[dependencies]
egobox-doe = { version = "0.x.y" }
egobox-gp = { version = "0.x.y" }
egobox-moe = { version = "0.x.y" }
egobox-ego = { version = "0.x.y" }
```

### Run the existing examples

```bash
cargo run -p egobox-ego --example xsinx --release
cargo run -p egobox-gp --example kriging --release
```

