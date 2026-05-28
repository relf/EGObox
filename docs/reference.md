# Reference

## Python package

- [PyPI package](https://pypi.org/project/egobox/)
- [Python package README](https://github.com/relf/egobox/blob/master/python/README.md)

## Rust crates

- [egobox-doe on docs.rs](https://docs.rs/egobox-doe)
- [egobox-gp on docs.rs](https://docs.rs/egobox-gp)
- [egobox-moe on docs.rs](https://docs.rs/egobox-moe)
- [egobox-ego on docs.rs](https://docs.rs/egobox-ego)

## Repository entry points

- [Root README](https://github.com/relf/egobox/blob/master/README.md)
- [Notebook index](https://github.com/relf/egobox/blob/master/doc/README.md)
- [Python examples directory](https://github.com/relf/egobox/tree/master/python/egobox/examples)
- [Rust crate directory](https://github.com/relf/egobox/tree/master/crates)

## Local docs build

```bash
conda activate base
python --version
python -m pip install -r requirements-docs.txt
python -m mkdocs build --strict
```