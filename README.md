<p align="center">
  <img
    width="100"
    src="./doc/LOGO_EGOBOX_v4_100x100.png"
    alt="Efficient Global Optimization toolbox in Rust"
  />
</p>

# EGObox - Efficient Global Optimization toolbox

![tests](https://github.com/relf/egobox/actions/workflows/test.yml/badge.svg)
![pytests](https://github.com/relf/egobox/actions/workflows/pytest.yml/badge.svg)
![linting](https://github.com/relf/egobox/actions/workflows/lint.yml/badge.svg)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04737/status.svg)](https://doi.org/10.21105/joss.04737)

Rust toolbox for Efficient Global Optimization method (arguably the most well-known bayesian optimization algorithm)
which adresses the gradient-free optimization of expensive objective functions.

The `egobox` package is twofold:

1. for end-users: [a Python module](#the-python-module), the Python binding of the optimizer named `Egor` and the surrogate model `Gpx`, mixture of Gaussian processes, written in Rust.
2. for developers: [a set of Rust libraries](#the-rust-libraries) useful to implement bayesian optimization (EGO-like) algorithms,

## The Python module

### Installation

```bash
pip install egobox
```

### Egor optimizer

```python
import numpy as np
import egobox as egx

# Objective function
def f_obj(x: np.ndarray) -> np.ndarray:
    return (x - 3.5) * np.sin((x - 3.5) / (np.pi))

# Minimize f_opt in [0, 25]
res = egx.Egor([[0.0, 25.0]], seed=42).minimize(f_obj, max_iters=20)
print(f"Optimization f={res.y_opt} at {res.x_opt}")  # Optimization f=[-15.12510323] at [18.93525454]
```

See also [this example written in Rust](crates/ego/examples/xsinx.rs)

### Gpx surrogate model

```python
import numpy as np
import matplotlib.pyplot as plt
import egobox as egx

# Training
xtrain = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
ytrain = np.array([0.0, 1.0, 1.5, 0.9, 1.0])
gpx = egx.Gpx.builder().fit(xtrain, ytrain)

# Prediction
xtest = np.linspace(0, 4, 100).reshape((-1, 1))
ytest = gpx.predict(xtest)

# Plot
plt.plot(xtest, ytest)
plt.plot(xtrain, ytrain, "o")
plt.show()
```

See also [this example written in Rust](crates/gp/examples/kriging.rs)

See the [tutorial notebooks](https://github.com/relf/egobox/tree/master/doc/README.md) and [examples folder](https://github.com/relf/egobox/tree/d9db0248199558f23d966796737d7ffa8f5de589/python/egobox/examples) for more information on the usage of the optimizer and mixture of Gaussian processes surrogate model.

## The Rust libraries

`egobox` Rust libraries consists of the following sub-packages.

| Name                                                         | Version                                                                                         | Documentation                                                               | Description                                                                               |
| :----------------------------------------------------------- | :---------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------- |
| [doe](https://github.com/relf/egobox/tree/master/crates/doe) | [![crates.io](https://img.shields.io/crates/v/egobox-doe)](https://crates.io/crates/egobox-doe) | [![docs](https://docs.rs/egobox-doe/badge.svg)](https://docs.rs/egobox-doe) | sampling methods; contains LHS, FullFactorial, Random methods                             |
| [gp](https://github.com/relf/egobox/tree/master/crates/gp)   | [![crates.io](https://img.shields.io/crates/v/egobox-gp)](https://crates.io/crates/egobox-gp)   | [![docs](https://docs.rs/egobox-gp/badge.svg)](https://docs.rs/egobox-gp)   | gaussian process regression; contains Kriging, PLS dimension reduction and sparse methods |
| [moe](https://github.com/relf/egobox/tree/master/crates/moe) | [![crates.io](https://img.shields.io/crates/v/egobox-moe)](https://crates.io/crates/egobox-moe) | [![docs](https://docs.rs/egobox-moe/badge.svg)](https://docs.rs/egobox-moe) | mixture of experts using GP models                                                        |
| [ego](https://github.com/relf/egobox/tree/master/crates/ego) | [![crates.io](https://img.shields.io/crates/v/egobox-ego)](https://crates.io/crates/egobox-ego) | [![docs](https://docs.rs/egobox-ego/badge.svg)](https://docs.rs/egobox-ego) | efficient global optimization with constraints and mixed integer handling                 |

### Usage

Depending on the sub-packages you want to use, you have to add following declarations to your `Cargo.toml`

```text
[dependencies]
egobox-doe = { version = "0.x.y" }
egobox-gp  = { version = "0.x.y" }
egobox-moe = { version = "0.x.y" }
egobox-ego = { version = "0.x.y" }
```

### Features

The table below presents the various features available depending on the subcrate

| Name         | doe  | gp   | moe  | ego  |
| :----------- | :--- | :--- | :--- | :--- |
| serializable | ✔️    | ✔️    | ✔️    |      |
| persistent   |      |      | ✔️    | ✔️(*) |
| blas         |      | ✔️    | ✔️    | ✔️    |
| nlopt        |      | ✔️    |      | ✔️    |

(*) for persistent mixture of gaussian processes with discrete variable available in `ego`

#### serializable

When selected, the serialization with [serde crate](https://serde.rs/) is enabled.

#### persistent

When selected, the save and load as a json file with [serde_json crate](https://serde.rs/) is enabled.

#### blas

When selected, the usage of BLAS/LAPACK backend is possible, see [below](#blaslapack-backend-optional) for more information.

#### nlopt

When selected, the [nlopt crate](https://github.com/adwhit/rust-nlopt) is used to provide optimizer implementations (ie Cobyla, Slsqp)

### Examples

Examples (in `examples/` sub-packages folder) are run as follows:

```bash
cd doe && cargo run --example samplings --release
```

``` bash
cd gp && cargo run --example kriging --release
```

``` bash
cd moe && cargo run --example clustering --release
```

``` bash
cd ego && cargo run --example ackley --release
```

### BLAS/LAPACK backend (optional)

`egobox` relies on [linfa](https://github.com/rust-ml/linfa) project for methods like clustering and dimension reduction, but also try to adopt as far as possible the same [coding structures](https://github.com/rust-ml/linfa/blob/master/CONTRIBUTE.md).

As for `linfa`, the linear algebra routines used in `gp`, `moe` ad `ego` are provided by the pure-Rust [linfa-linalg](https://github.com/rust-ml/linfa-linalg) crate, the default linear algebra provider.

Otherwise, you can choose an external BLAS/LAPACK backend available through the [ndarray-linalg](https://github.com/rust-ndarray/ndarray-linalg) crate. In this case, you have to specify the `blas` feature and a `linfa` [BLAS/LAPACK backend feature](https://github.com/rust-ml/linfa#blaslapack-backend) (more information in [linfa features](https://github.com/rust-ml/linfa#blaslapack-backend)).

Thus, for instance, to use `gp` with the Intel MKL BLAS/LAPACK backend, you could specify in your `Cargo.toml` the following features:

```text
[dependencies]
egobox-gp = { version = "0.x.y", features = ["blas", "linfa/intel-mkl-static"] }
```

or you could run the `gp` example as follows:

``` bash
cd gp && cargo run --example kriging --release --features blas,linfa/intel-mkl-static
```

## Citation

[![DOI](https://joss.theoj.org/papers/10.21105/joss.04737/status.svg)](https://doi.org/10.21105/joss.04737)

If you find this project useful for your research, you may cite it as follows:

```text
@article{
  Lafage2022, 
  author = {Rémi Lafage}, 
  title = {egobox, a Rust toolbox for efficient global optimization}, 
  journal = {Journal of Open Source Software} 
  year = {2022}, 
  doi = {10.21105/joss.04737}, 
  url = {https://doi.org/10.21105/joss.04737}, 
  publisher = {The Open Journal}, 
  volume = {7}, 
  number = {78}, 
  pages = {4737}, 
} 
```

Additionally, you may consider adding a star to the repository. This positive feedback improves the visibility of the project.

## References

Bartoli, N., Lefebvre, T., Dubreuil, S., Olivanti, R., Priem, R., Bons, N., Martins, J. R. R. A., & Morlier, J. (2019).
[Adaptive modeling strategy for constrained global optimization with application to aerodynamic wing design](https://doi.org/10.1016/j.ast.2019.03.041).
Aerospace Science and Technology, 90, 85–102.

Bouhlel, M. A., Bartoli, N., Otsmane, A., & Morlier, J. (2016).
[Improving kriging surrogates of high-dimensional design models by partial least squares dimension reduction](https://doi.org/10.1007/s00158-015-1395-9).
Structural and Multidisciplinary Optimization, 53(5), 935–952.

Bouhlel, M. A., Hwang, J. T., Bartoli, N., Lafage, R., Morlier, J., & Martins, J. R. R. A.
(2019). [A python surrogate modeling framework with derivatives](https://doi.org/10.1016/j.advengsoft.2019.03.005).
Advances in Engineering Software, 102662.

Dubreuil, S., Bartoli, N., Gogu, C., & Lefebvre, T. (2020).
[Towards an efficient global multi-disciplinary design optimization algorithm](https://doi.org/10.1007/s00158-020-02514-6)
Structural and Multidisciplinary Optimization, 62(4), 1739–1765.

Jin, R. and Chen, W. and Sudjianto, A. (2005), 
[An efficient algorithm for constructing optimal design of computer experiments](https://doi.org/10.1016/j.jspi.2004.02.01) Journal of Statistical Planning and Inference, 134:268-287.

Bettebghor, Dimitri, et al. 
[Surrogate modeling approximation using a mixture of experts based on EM joint estimation](https://hal.archives-ouvertes.fr/hal-01852300/document)
Structural and multidisciplinary optimization 43.2 (2011): 243-259.

Jones, D. R., Schonlau, M., & Welch, W. J. (1998).
[Efficient global optimization of expensive black-box functions. Journal of Global Optimization](https://www.researchgate.net/publication/235709802_Efficient_Global_Optimization_of_Expensive_Black-Box_Functions), 13(4), 455–492.

Diouane, Youssef, et al. [TREGO: a trust-region framework for efficient global optimization](https://arxiv.org/pdf/2101.06808).
Journal of Global Optimization 86.1 (2023): 1-23.

Priem, Rémy, Nathalie Bartoli, and Youssef Diouane.
[On the use of upper trust bounds in constrained Bayesian optimization infill criteria](https://hal.science/hal-02182492v1/file/Priem_24049.pdf).
AIAA aviation 2019 forum. 2019.

Sasena M., Papalambros P., Goovaerts P., 2002.
[Global optimization of problems with disconnected feasible regions via surrogate modeling](https://deepblue.lib.umich.edu/handle/2027.42/77089).
AIAA Paper.

Ginsbourger, D., Le Riche, R., & Carraro, L. (2010).
[Kriging is well-suited to parallelize optimization](https://www.researchgate.net/publication/226716412_Kriging_Is_Well-Suited_to_Parallelize_Optimization)

E.C. Garrido-Merchan and D. Hernandez-Lobato.
[Dealing with categorical and integer-valued variables in Bayesian Optimization with Gaussian processes](https://arxiv.org/pdf/1805.03463).

Zhan, Dawei, et al. 
[A cooperative approach to efficient global optimization](https://link.springer.com/article/10.1007/s10898-023-01316-6). Journal of Global Optimization 88.2 (2024): 327-357

Lisa Pretsch et al.
[Bayesian optimization of cooperative components for multi-stage aero-structural compressor blade design](https://www.researchgate.net/publication/391492598_Bayesian_optimization_of_cooperative_components_for_multi-stage_aero-structural_compressor_blade_design). Struct Multidisc Optim 68, 84 (2025)

Appriou, T., Rullière, D. & Gaudrie, D,
[Combination of optimization-free kriging models for high-dimensional problems](https://doi.org/10.1007/s00180-023-01424-7),
Comput Stat 39, 3049–3071 (2024).

S. Ament, S. Daulton, D. Eriksson, M. Balandat, E. Bakshy,
[Unexpected improvements to expected improvement for bayesian optimization](https://arxiv.org/pdf/2310.20708),
Advances in Neural Information Processing Systems, 2023

Marrel, A., & Iooss, B. (2024).
[Probabilistic surrogate modeling by Gaussian process: A review on recent insights in estimation and validation.](https://www.sciencedirect.com/science/article/pii/S0951832024001686?casa_token=l-YuUssKYfsAAAAA:WpEZkMh7bf0Ago2rDi0I_E5BydKDTmjluPOP0KfAratwka89IRqu09U1J6OnpG1gc_8FIafRzA)
Reliability Engineering & System Safety, 247, 110094.

Bussemaker, Jasper H., et al. (2024)
[Surrogate-based optimization of system architectures subject to hidden constraints.](https://hal.science/hal-04462408v2/file/DTIS2024-018-DTIS2024-018%20posprint-Accept%C3%A9e.pdf)
AIAA AVIATION FORUM AND ASCEND 2024.

smtorg. (2018). [Surrogate modeling toolbox](https://github.com/SMTOrg/smt). GitHub.

## License

Licensed under the Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0>
