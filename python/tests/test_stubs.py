import ast
import inspect
import unittest
from pathlib import Path

import egobox as egx

STUB_PATH = Path(egx.__file__).parent / "egobox.pyi"
DICT_ANY = "builtins.dict[builtins.str, typing.Any]"


def _stub_classes():
    tree = ast.parse(STUB_PATH.read_text())
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def _stub_functions(body):
    """Public methods and constructor of a stub class body, properties excluded"""
    return {
        node.name: node
        for node in body
        if isinstance(node, ast.FunctionDef)
        and (node.name == "__new__" or not node.name.startswith("_"))
        and not any(
            ast.unparse(d).endswith("setter") or ast.unparse(d) == "property"
            for d in node.decorator_list
        )
    }


def _is_enum(cls: ast.ClassDef) -> bool:
    return any(ast.unparse(base) == "enum.Enum" for base in cls.bases)


def _param_names(fn: ast.FunctionDef):
    names = [a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs]
    return [n for n in names if n not in ("self", "cls")]


def _runtime_param_names(obj):
    return [n for n in inspect.signature(obj).parameters if n not in ("self", "cls")]


class TestStubs(unittest.TestCase):
    def test_enum_members_match_runtime(self):
        for name, cls in _stub_classes().items():
            if not _is_enum(cls):
                continue
            stub_members = {
                target.id
                for node in cls.body
                if isinstance(node, ast.Assign)
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            runtime_cls = getattr(egx, name)
            runtime_members = {
                attr
                for attr in dir(runtime_cls)
                if not attr.startswith("_")
                and isinstance(getattr(runtime_cls, attr), runtime_cls)
            }
            self.assertEqual(stub_members, runtime_members, f"enum {name}")

    def test_configs_have_constructor_in_stub(self):
        classes = _stub_classes()
        for name in ("GpConfig", "QEiConfig", "TregoConfig"):
            self.assertIn("__new__", _stub_functions(classes[name].body), name)

    def test_signatures_match_runtime(self):
        # stubtest cannot compare these signatures because pyo3 renders defaults as `...`
        # at runtime, so at least check that the parameter names are in sync
        for name, cls in _stub_classes().items():
            if _is_enum(cls):
                continue
            runtime_cls = getattr(egx, name)
            for fname, fn in _stub_functions(cls.body).items():
                runtime_fn = (
                    runtime_cls if fname == "__new__" else getattr(runtime_cls, fname)
                )
                self.assertEqual(
                    _param_names(fn),
                    _runtime_param_names(runtime_fn),
                    f"{name}.{fname}",
                )
        tree = ast.parse(STUB_PATH.read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                self.assertEqual(
                    _param_names(node),
                    _runtime_param_names(getattr(egx, node.name)),
                    node.name,
                )

    def test_no_any_in_public_signatures(self):
        tree = ast.parse(STUB_PATH.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            annotations = [a.annotation for a in node.args.args + node.args.kwonlyargs]
            annotations.append(node.returns)
            for ann in annotations:
                if ann is None:
                    continue
                text = ast.unparse(ann).replace(DICT_ANY, "")
                self.assertNotIn("typing.Any", text, f"{node.name}: {ast.unparse(ann)}")


if __name__ == "__main__":
    unittest.main()
