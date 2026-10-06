# EGObox 0.38.1 release notes

Bug-fix release, mainly for mixed-integer problems. No API changes.

* **Mixed-integer enum folding/unfolding** (911f248, 3a04e5a): wrong columns were read when an enum followed
  another enum or a non-enum variable followed an enum.
* **Function constraints with mixed-integer inputs** (6534cce): `fcstrs` now receive `x` in folded discrete
  space, and their gradient is mapped back to the continuous relaxed space.
* **`Egor` panic when the objective fails at every initial point** (320cf70): `run()` now returns an
  `ObjectiveFunctionError` (`RuntimeError` in Python).
* **Mixed-integer sampling fairness** (d79f269, aba35ac): integer, ordered and enum levels are sampled with
  equal probability (bounds were under-sampled). Applies to `MixintContext` LHS/random, Python
  `lhs`/`sampling` and `Egor` initial DOE. Results differ from 0.38.0 for a given seed.
