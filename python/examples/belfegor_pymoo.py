# =====================================================
# Egobox demo: multi-objective optimization of pymoo test problems with Belfegor
# =====================================================
#
# Approximate the Pareto front of a pymoo (https://pymoo.org) multi-objective test problem
# with Belfegor, then plot the front and print quality metrics wrt the true front.
#
# Requires pymoo (`pip install pymoo`). Examples:
#
#   python belfegor_pymoo.py --list
#   python belfegor_pymoo.py zdt1 --n-var 5
#   python belfegor_pymoo.py bnh --strategy eim --max-iters 30
#   python belfegor_pymoo.py dtlz2 --n-var 5 --n-obj 3 --strategy qehvi --batch 3
#   python belfegor_pymoo.py wfg1 --n-var 6 --n-obj 2
#   python belfegor_pymoo.py dascmop1 --n-var 5 --arg difficulty=1
#

import argparse
import ast
import inspect
import re
import sys
import time
import warnings

import numpy as np

import egobox as egx

try:
    from pymoo.indicators.gd import GD
    from pymoo.indicators.hv import HV
    from pymoo.indicators.igd import IGD
    from pymoo.indicators.igd_plus import IGDPlus
    from pymoo.problems import get_problem
    from pymoo.util.ref_dirs import get_reference_directions
except ImportError:
    sys.exit("pymoo is required to run this example: pip install pymoo")

STRATEGIES = {
    "auto": None,
    "parego": egx.MooStrategy.PAREGO,
    "eim": egx.MooStrategy.EIM,
    "ehvi": egx.MooStrategy.EHVI,
    "qehvi": egx.MooStrategy.QEHVI,
}

EIM_AGGREGATIONS = {
    "euclidean": egx.EimAggregation.EUCLIDEAN,
    "maximin": egx.EimAggregation.MAXIMIN,
    "hypervolume": egx.EimAggregation.HYPERVOLUME,
}

# Multi-objective problems of pymoo 0.6, used when the registry can not be read
DEFAULT_PROBLEMS = (
    ["bnh", "carside", "kursawe", "osy", "srn", "tnk", "truss2d", "welded_beam"]
    + [f"ctp{i}" for i in range(1, 9)]
    + [f"zdt{i}" for i in range(1, 7)]
    + [f"dtlz{i}" for i in range(1, 8)]
    + [f"wfg{i}" for i in range(1, 10)]
    + [f"mw{i}" for i in range(1, 15)]
    + [f"dascmop{i}" for i in range(1, 10)]
)


# -----------------------------------------------------
# pymoo problems
# -----------------------------------------------------
def registered_problems():
    """Names of the problems known by pymoo get_problem()"""
    # pymoo keeps its registry as a dict literal inside get_problem()
    names = re.findall(
        r'^\s+"([a-z0-9_\-]+)":', inspect.getsource(get_problem), re.MULTILINE
    )
    return names or DEFAULT_PROBLEMS


def list_problems():
    """Print the multi-objective problems: name, number of variables, objectives, constraints"""
    print(f"{'problem':<16} {'n_var':>5} {'n_obj':>5} {'n_cstr':>6}")
    requiring_args = []
    for name in registered_problems():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                problem = get_problem(name)
        except Exception:  # noqa: BLE001 (pymoo raises plain exceptions)
            requiring_args.append(name)
            continue
        if problem.n_obj < 2:
            continue  # single-objective problem
        n_cstr = problem.n_ieq_constr + problem.n_eq_constr
        print(f"{name:<16} {problem.n_var:>5} {problem.n_obj:>5} {n_cstr:>6}")
    if requiring_args:
        print(
            "\nRequiring arguments (e.g. --n-var 6 --n-obj 2 for wfg*, "
            "--arg difficulty=1 for dascmop*): " + ", ".join(requiring_args)
        )


def parse_args_kv(items):
    """Problem constructor arguments given as KEY=VALUE strings"""
    kwargs = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            sys.exit(f"--arg expects KEY=VALUE, got {item!r}")
        try:
            kwargs[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            kwargs[key] = value
    return kwargs


def make_problem(args):
    kwargs = parse_args_kv(args.arg)
    if args.n_var is not None:
        kwargs["n_var"] = args.n_var
    if args.n_obj is not None:
        kwargs["n_obj"] = args.n_obj
    try:
        problem = get_problem(args.problem, **kwargs)
    except Exception as err:  # noqa: BLE001 (pymoo raises plain exceptions)
        sys.exit(f"Can not create pymoo problem {args.problem!r} with {kwargs}: {err}")
    if problem.n_obj < 2:
        sys.exit(f"{args.problem} is a single-objective problem, use Egor instead")
    return problem


def true_front(problem):
    """True Pareto front of the problem if pymoo provides it, None otherwise"""
    # pymoo raises plain exceptions when the front is not available
    try:
        return problem.pareto_front()
    except Exception:  # noqa: BLE001
        # many-objective problems need reference directions
        ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=6)
    try:
        return problem.pareto_front(ref_dirs=ref_dirs)
    except Exception as err:  # noqa: BLE001
        print(f"True Pareto front not available: {err}")
        return None


def objective_function(problem):
    """Belfegor objective function [F, G, H] and the corresponding constraint specs"""

    def fun(x):
        F, G, H = problem.evaluate(x, return_values_of=["F", "G", "H"])
        columns = [np.atleast_2d(F)]
        if problem.n_ieq_constr > 0:
            columns.append(np.atleast_2d(G))
        if problem.n_eq_constr > 0:
            columns.append(np.atleast_2d(H))
        return np.hstack(columns).astype(np.float64)

    # pymoo inequality constraints are G <= 0 like egobox ones, equality constraints are H = 0
    cstr_specs = [egx.CstrSpec.leq(0.0)] * problem.n_ieq_constr + [
        egx.CstrSpec.eq(0.0)
    ] * problem.n_eq_constr
    return fun, cstr_specs


# -----------------------------------------------------
# Optimization
# -----------------------------------------------------
def run(problem, args):
    fun, cstr_specs = objective_function(problem)
    xspecs = np.column_stack([problem.xl, problem.xu]).astype(np.float64)
    moo_config = egx.MooConfig(
        strategy=STRATEGIES[args.strategy],
        batch=args.batch,
        eim_aggregation=EIM_AGGREGATIONS[args.eim_aggregation],
        hv_stop=tuple(args.hv_stop) if args.hv_stop else None,
    )
    belfegor = egx.Belfegor(
        xspecs,
        n_obj=problem.n_obj,
        moo_config=moo_config,
        cstr_specs=cstr_specs or None,
        n_doe=args.n_doe,
        seed=args.seed,
    )
    print(
        f"Optimize {args.problem} (n_var={problem.n_var}, n_obj={problem.n_obj}, "
        f"n_cstr={len(cstr_specs)}) with {moo_config}..."
    )
    start = time.perf_counter()
    res = belfegor.minimize(fun, max_iters=args.max_iters, verbose=args.verbose)
    return res, time.perf_counter() - start


def reference_point(front):
    """Front nadir point plus 10% of its range"""
    ideal, nadir = front.min(axis=0), front.max(axis=0)
    span = np.where(nadir - ideal > 0.0, nadir - ideal, 1.0)
    return nadir + 0.1 * span


def metrics(res, problem, pf, elapsed):
    """Quality metrics of the found front (objective values only)"""
    found = res.y_pareto[:, : problem.n_obj]
    values = {
        "evaluations": len(res.y_doe),
        "front size": len(found),
        "exit status": str(res.status.exit),
        "time (s)": round(elapsed, 2),
    }
    if pf is not None:
        ref_point = reference_point(pf)
        hv = HV(ref_point=ref_point)(found)
        hv_true = HV(ref_point=ref_point)(pf)
        values["HV (ref: true front nadir + 10%)"] = hv
        values["HV of the true front"] = hv_true
        values["HV ratio"] = hv / hv_true if hv_true > 0.0 else float("nan")
        values["IGD"] = IGD(pf)(found)
        values["IGD+"] = IGDPlus(pf)(found)
        values["GD"] = GD(pf)(found)
    else:
        values["HV (ref: found front nadir + 10%)"] = HV(
            ref_point=reference_point(found)
        )(found)
        values["true front"] = "not available"
    return values


def print_metrics(values):
    width = max(len(k) for k in values)
    print()
    for key, value in values.items():
        if isinstance(value, float):
            value = f"{value:.6g}"
        print(f"{key:<{width}} : {value}")


# -----------------------------------------------------
# Plot
# -----------------------------------------------------
def plot(res, problem, pf, values, args):
    import matplotlib

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_obj = problem.n_obj
    found = res.y_pareto[:, :n_obj]
    evaluated = res.y_doe[:, :n_obj]
    compromise = res.y_opt[:n_obj]
    title = f"{args.problem} with Belfegor ({args.strategy})"
    if "HV ratio" in values:
        title += f"\nHV ratio={values['HV ratio']:.3f}, IGD={values['IGD']:.3g}"

    if n_obj == 2:
        fig, ax = plt.subplots(figsize=(7, 6))
        if pf is not None:
            order = np.argsort(pf[:, 0])
            ax.plot(pf[order, 0], pf[order, 1], "k.", ms=2, label="true front")
        ax.plot(
            evaluated[:, 0], evaluated[:, 1], ".", color="grey", label="evaluations"
        )
        ax.plot(found[:, 0], found[:, 1], "o", mfc="none", label="Pareto front")
        ax.plot(compromise[0], compromise[1], "r*", ms=14, label="compromise")
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.legend()
    elif n_obj == 3:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(projection="3d")
        if pf is not None:
            ax.scatter(*pf.T, s=2, c="k", alpha=0.3, label="true front")
        ax.scatter(*evaluated.T, s=12, c="grey", marker="x", label="evaluations")
        ax.scatter(*found.T, s=30, label="Pareto front")
        ax.scatter(*compromise, s=150, c="r", marker="*", label="compromise")
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.set_zlabel("f3")
        ax.legend()
    else:
        # parallel coordinates of the front, objectives normalized with the front bounds
        fig, ax = plt.subplots(figsize=(9, 6))
        lower, upper = found.min(axis=0), found.max(axis=0)
        span = np.where(upper - lower > 0.0, upper - lower, 1.0)
        axes = np.arange(1, n_obj + 1)
        for y in (found - lower) / span:
            ax.plot(axes, y, color="tab:blue", alpha=0.6)
        ax.plot(axes, (compromise - lower) / span, "r-", lw=3, label="compromise")
        ax.set_xticks(axes, [f"f{i}" for i in axes])
        ax.set_ylabel("normalized objective value (front bounds)")
        ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f"Plot saved to {args.save}")
    if not args.no_show:
        plt.show()


# -----------------------------------------------------
# Command line
# -----------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Approximate the Pareto front of a pymoo multi-objective test problem with Belfegor"
    )
    parser.add_argument("problem", nargs="?", help="pymoo problem name (see --list)")
    parser.add_argument(
        "--list", action="store_true", help="list the multi-objective problems"
    )
    parser.add_argument("--n-var", type=int, help="number of variables of the problem")
    parser.add_argument(
        "--n-obj", type=int, help="number of objectives (scalable problems)"
    )
    parser.add_argument(
        "--arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="other problem constructor argument (repeatable), e.g. difficulty=1",
    )
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES),
        default="auto",
        help="multi-objective strategy (auto: EHVI for 2-3 objectives, ParEGO beyond)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="points evaluated per iteration (at most 4 with qehvi)",
    )
    parser.add_argument(
        "--eim-aggregation",
        choices=list(EIM_AGGREGATIONS),
        default="euclidean",
        help="aggregation of the EIM strategy",
    )
    parser.add_argument(
        "--n-doe", type=int, default=0, help="initial DOE size (0: automatic)"
    )
    parser.add_argument("--max-iters", type=int, default=20, help="iteration budget")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--hv-stop",
        nargs=2,
        type=float,
        metavar=("TOL", "N_ITERS"),
        help="stop when the front hypervolume improves by less than TOL over N_ITERS iterations",
    )
    parser.add_argument(
        "--verbose", type=int, default=0, help="Belfegor verbosity (0 to 4)"
    )
    parser.add_argument("--save", metavar="FILE", help="save the plot to FILE")
    parser.add_argument(
        "--no-show", action="store_true", help="do not display the plot window"
    )
    args = parser.parse_args(argv)

    if args.list:
        list_problems()
        return
    if args.problem is None:
        parser.error("a problem name is required (see --list)")
    if args.hv_stop:
        args.hv_stop = (args.hv_stop[0], int(args.hv_stop[1]))

    problem = make_problem(args)
    res, elapsed = run(problem, args)
    pf = true_front(problem)
    values = metrics(res, problem, pf, elapsed)
    print_metrics(values)
    plot(res, problem, pf, values, args)


if __name__ == "__main__":
    main()
