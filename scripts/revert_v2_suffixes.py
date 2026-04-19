#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
TSV_PATH = ROOT / "data" / "research_list.tsv"
ARCHIVE_DIR = ROOT / "data" / "archives"

TARGET_COLUMN = "ソースのファイル名"


def read_tsv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("TSVヘッダを読めませんでした。")
        headers = list(reader.fieldnames)
        rows = list(reader)
    return rows, headers


def write_tsv(path: Path, rows: List[Dict[str, str]], headers: List[str]) -> None:
    tmp_path = path.with_suffix(".tmp.tsv")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=headers,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    shutil.move(str(tmp_path), str(path))


def backup_tsv(path: Path) -> Path:
    backup_path = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup_path)
    return backup_path


def split_filenames(cell_value: str) -> List[str]:
    if not cell_value:
        return []
    parts: List[str] = []
    for line in str(cell_value).splitlines():
        line = line.strip()
        if not line:
            continue
        if ", " in line:
            parts.extend([p.strip() for p in line.split(", ") if p.strip()])
        else:
            parts.append(line)
    return parts


def join_filenames(names: List[str]) -> str:
    return ", ".join(names)


def remove_v2_suffix(name: str) -> str:
    """
    foo_01_v2.pdf -> foo_01.pdf
    foo_v2.docx   -> foo.docx
    """
    return re.sub(r"_v2(\.[^.]+)$", r"\1", name)


def uniquify_target(path: Path) -> Path:
    """
    戻し先が既に存在してしまう場合は安全のためそのまま返さず None 相当に扱いたいが、
    ここでは衝突回避のため _restored2 を付ける。
    """
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    n = 2
    while True:
        candidate = parent / f"{stem}_restored{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="_v2 を外してファイル名とTSVを戻す")
    parser.add_argument("--tsv", type=Path, default=TSV_PATH)
    parser.add_argument("--archive-dir", type=Path, default=ARCHIVE_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tsv_path = args.tsv.resolve()
    archive_dir = args.archive_dir.resolve()

    rows, headers = read_tsv(tsv_path)
    if TARGET_COLUMN not in headers:
        raise ValueError(f"{TARGET_COLUMN!r} 列がありません")

    print(f"TSV: {tsv_path}")
    print(f"ARCHIVE_DIR: {archive_dir}")
    print(f"DRY_RUN: {args.dry_run}")
    print()

    changed_rows = 0

    for idx, row in enumerate(rows, start=1):
        original = row.get(TARGET_COLUMN, "") or ""
        names = split_filenames(original)
        if not names:
            continue

        new_names: List[str] = []
        row_changed = False

        print(f"--- row {idx}")

        for name in names:
            reverted = remove_v2_suffix(name)

            if reverted == name:
                new_names.append(name)
                continue

            src = archive_dir / name
            dst = archive_dir / reverted

            if not src.exists():
                print(f"MISS_SOURCE: {src}")
                new_names.append(name)
                continue

            final_dst = uniquify_target(dst)
            print(f"RENAME: {src.name} -> {final_dst.name}")

            if not args.dry_run:
                src.rename(final_dst)

            new_names.append(final_dst.name)
            row_changed = True

        if row_changed:
            row[TARGET_COLUMN] = join_filenames(new_names)
            changed_rows += 1
            print(f"UPDATE_TSV: {original} -> {row[TARGET_COLUMN]}")
        else:
            print("NO_CHANGE")

        print()

    if args.dry_run:
        print("dry-run のため、変更していません。")
        return

    backup_path = backup_tsv(tsv_path)
    write_tsv(tsv_path, rows, headers)

    print(f"完了: {changed_rows} 行更新")
    print(f"TSVバックアップ: {backup_path}")
    print(f"更新後TSV: {tsv_path}")


if __name__ == "__main__":
    main()