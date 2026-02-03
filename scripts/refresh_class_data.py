#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from pathlib import Path


def clear_directory(target_dir: Path) -> None:
    for entry in target_dir.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def resolve_config_path(value: str | None, repo_root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"Failed to read config {config_path}: {exc}", file=sys.stderr)
        return {}


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "scripts" / "class_data_sources.json",
        help="Path to class-data source config JSON.",
    )
    config_args, remaining_args = config_parser.parse_known_args()

    config = load_config(config_args.config)
    default_class_public_root = repo_root.parent / "class_public"
    default_docs_root = repo_root.parent / "class-code-documentation"

    class_public_root = (
        resolve_config_path(config.get("class_public_path"), repo_root)
        or default_class_public_root
    )
    class_docs_root = (
        resolve_config_path(config.get("class_code_documentation_path"), repo_root)
        or default_docs_root
    )

    parser = argparse.ArgumentParser(
        description=(
            "Refresh class-data from CLASS scripts and documentation .rst files."
        ),
        parents=[config_parser],
    )
    parser.add_argument(
        "--class-data",
        type=Path,
        default=repo_root / "class-data",
        help="Destination class-data directory.",
    )
    parser.add_argument(
        "--class-scripts",
        type=Path,
        default=class_public_root / "scripts",
        help="Source CLASS scripts directory.",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=class_docs_root / "docs",
        help="Source CLASS documentation docs directory.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep existing class-data files instead of cleaning first.",
    )
    args = parser.parse_args(remaining_args)

    missing_sources = [
        path for path in (args.class_scripts, args.docs_dir) if not path.exists()
    ]
    if missing_sources:
        print("Missing source directories:", file=sys.stderr)
        for path in missing_sources:
            print(f"- {path}", file=sys.stderr)
        return 1

    print("Using sources:")
    print(f"- CLASS scripts: {args.class_scripts}")
    print(f"- CLASS docs:    {args.docs_dir}")
    print(f"- class-data:    {args.class_data}")

    args.class_data.mkdir(parents=True, exist_ok=True)
    if not args.keep_existing:
        clear_directory(args.class_data)

    py_files = sorted(args.class_scripts.glob("*.py"))
    rst_files = sorted(args.docs_dir.glob("*.rst"))

    for source in py_files + rst_files:
        shutil.copy2(source, args.class_data / source.name)

    print(
        f"Copied {len(py_files)} .py files and {len(rst_files)} .rst files to"
        f" {args.class_data}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
