from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

DEFAULT_SETTINGS_FILE = Path("src/config/settings.py")
DEFAULT_OUTPUT_FILE = Path(".env.example")
SECRET_FIELD_MARKERS = ("api_key", "secret", "token", "password")


def _annotation_contains_secret(node: ast.AST | None) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.Name):
        return node.id == "SecretStr"
    if isinstance(node, ast.Attribute):
        return node.attr == "SecretStr"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _annotation_contains_secret(node.left) or _annotation_contains_secret(
            node.right
        )
    if isinstance(node, ast.Subscript):
        return _annotation_contains_secret(node.value) or _annotation_contains_secret(
            node.slice
        )
    if isinstance(node, ast.Tuple | ast.List):
        return any(_annotation_contains_secret(item) for item in node.elts)
    return False


def _literal_to_env_value(
    node: ast.AST | None,
    field_name: str,
    annotation: ast.AST | None = None,
) -> str:
    if (
        node is None
        or _annotation_contains_secret(annotation)
        or any(marker in field_name.lower() for marker in SECRET_FIELD_MARKERS)
    ):
        return ""

    if isinstance(node, ast.Constant):
        value = node.value
        if value is None:
            return ""
        if isinstance(value, bool):
            return str(value).lower()
        return str(value)

    if isinstance(node, ast.Call):
        func_name = _call_name(node.func)
        if func_name == "Path" and node.args:
            return _literal_to_env_value(node.args[0], field_name, annotation)
        if func_name == "Field":
            return _field_default_to_env_value(node, field_name, annotation)

    try:
        value = ast.literal_eval(node)
    except (TypeError, ValueError):
        return ""

    return str(value)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _field_default_to_env_value(
    node: ast.Call,
    field_name: str,
    annotation: ast.AST | None,
) -> str:
    for keyword in node.keywords:
        if keyword.arg == "default":
            return _literal_to_env_value(keyword.value, field_name, annotation)
        if keyword.arg == "default_factory":
            return ""
    if node.args:
        return _literal_to_env_value(node.args[0], field_name, annotation)
    return ""


def _field_description(node: ast.AST | None) -> str:
    if not isinstance(node, ast.Call) or _call_name(node.func) != "Field":
        return ""

    for keyword in node.keywords:
        if keyword.arg == "description" and isinstance(keyword.value, ast.Constant):
            value = keyword.value.value
            return value if isinstance(value, str) else ""
    return ""


def build_env_example(settings_file: Path) -> str:
    tree = ast.parse(settings_file.read_text(encoding="utf-8"))
    settings_class = _find_settings_class(tree)

    lines = ["# ====== 应用配置 ======"]
    for node in settings_class.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue

        field_name = node.target.id
        if field_name == "model_config":
            continue

        env_name = field_name.upper()
        env_value = _literal_to_env_value(node.value, field_name, node.annotation)
        description = _field_description(node.value)
        if description:
            lines.append(f"# {description}")
        lines.append(f"{env_name}={env_value}")

    return "\n".join(lines) + "\n"


def _find_settings_class(tree: ast.Module) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "EnvSettings":
            return node
    raise RuntimeError("未在配置文件中找到 Settings 类。")


def write_env_example(settings_file: Path, output_file: Path) -> bool:
    content = build_env_example(settings_file)
    if output_file.exists() and output_file.read_text(encoding="utf-8") == content:
        return False
    output_file.write_text(content, encoding="utf-8")
    return True


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "根据 src/python_project_boilerplate/config/settings.py 生成 .env.example，"
            "不会读取本地 .env。"
        )
    )
    parser.add_argument("--settings-file", type=Path, default=DEFAULT_SETTINGS_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    content = build_env_example(args.settings_file)

    if args.check:
        current = (
            args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        )
        if current == content:
            return 0
        print(f"{args.output} is out of date. Run `make env-example`.", file=sys.stderr)
        return 1

    changed = write_env_example(args.settings_file, args.output)
    if changed:
        print(f"Updated {args.output}")
    else:
        print(f"{args.output} is already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
