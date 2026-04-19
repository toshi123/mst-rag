#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import re
import shutil
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple, Optional


ROOT = Path(__file__).resolve().parents[1]
TSV_PATH = ROOT / "data" / "research_list.tsv"
ARCHIVE_DIR = ROOT / "data" / "archives"

REQUIRED_COLUMNS = [
    "著者名",
    "研究タイトル",
    "ソースのファイル名",
    "提出年度",
    "形式",
]

ALLOWED_FORMATS = {"Title", "Poster", "Report"}

# すでに整っているファイル名の判定（先頭の西暦は ASCII 数字4桁のみ）
# Python の \\d は全角数字などもマッチするため [0-9]{4} に固定する。
# 例: 2025_Report_田中太郎_研究タイトル_01.pdf
# ２０２５_Report_... のように全角年のみの名前は未統一とみなしリネーム対象にする。
ALREADY_RENAMED_PATTERN = re.compile(
    r"^[0-9]{4}_(Title|Poster|Report)_.+?_[0-9]{2}\.[^.]+$"
)


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return unicodedata.normalize("NFKC", str(text)).strip()


def normalize_for_match(text: str) -> str:
    if text is None:
        return ""
    text = str(text).replace("＿", "_")
    text = unicodedata.normalize("NFKC", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def sanitize_filename(text: str, max_len: int = 80) -> str:
    text = normalize_text(text)
    text = re.sub(r'[\\/:*?"<>|]', "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("._ ")
    if not text:
        text = "untitled"
    return text[:max_len]


def split_filenames(cell_value: str) -> List[str]:
    """
    区切りは ', ' と改行のみ。
    '、' は区切りとして扱わない。
    """
    value = normalize_text(cell_value)
    if not value:
        return []

    lines = [v.strip() for v in value.splitlines() if v.strip()]
    parts: List[str] = []

    for line in lines if lines else [value]:
        if ", " in line:
            parts.extend([p.strip() for p in line.split(", ") if p.strip()])
        else:
            parts.append(line.strip())

    return [p for p in parts if p]


def extract_first_author(author_text: str) -> str:
    text = normalize_text(author_text)
    if not text:
        return "UnknownAuthor"

    first = re.split(r"[,、;]\s*", text)[0]
    first = re.sub(r"\(.*?\)", "", first)
    first = re.sub(r"（.*?）", "", first)
    first = first.strip()

    first = sanitize_filename(first, max_len=30)
    return first or "UnknownAuthor"


def normalize_format(fmt: str) -> str:
    fmt = normalize_text(fmt)
    if fmt not in ALLOWED_FORMATS:
        raise ValueError(
            f"形式 が不正です: {fmt!r} (許可値: {sorted(ALLOWED_FORMATS)})"
        )
    return fmt


def build_new_filename(
    year: str,
    fmt: str,
    first_author: str,
    title: str,
    index: int,
    suffix: str,
) -> str:
    year_part = sanitize_filename(year, max_len=10) or "untitled"
    fmt_part = sanitize_filename(fmt, max_len=10)
    author_part = sanitize_filename(first_author, max_len=30)
    title_part = sanitize_filename(title, max_len=100)
    return f"{year_part}_{fmt_part}_{author_part}_{title_part}_{index:02d}{suffix.lower()}"


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


def validate_headers(headers: List[str]) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in headers]
    if missing:
        raise ValueError(f"TSVに必要な列がありません: {missing}")


def find_archive_file(archive_dir: Path, old_name: str) -> Optional[Path]:
    old_name = old_name.strip()

    direct = archive_dir / old_name
    if direct.exists() and direct.is_file():
        return direct

    target = normalize_for_match(old_name)

    for p in archive_dir.iterdir():
        if p.is_file() and normalize_for_match(p.name) == target:
            return p

    return None


def looks_like_non_file_value(name: str) -> bool:
    name = normalize_text(name)
    if not name:
        return True
    if "." not in name:
        return True
    return False


def is_already_renamed(name: str) -> bool:
    return bool(ALREADY_RENAMED_PATTERN.match(name))


def filename_has_untitled_placeholder(name: str) -> bool:
    """
    欠損メタで sanitize により付いた untitled を含むファイル名は、
    TSV を埋めたあとに改めてリネームできる対象とする。
    """
    return "untitled" in normalize_text(name).lower()


def backup_tsv(path: Path) -> Path:
    backup_path = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup_path)
    return backup_path


def rename_files_for_row(
    row: Dict[str, str],
    archive_dir: Path,
    dry_run: bool = True,
) -> Tuple[str, List[str]]:
    logs: List[str] = []

    author_text = normalize_text(row.get("著者名", ""))
    title = normalize_text(row.get("研究タイトル", ""))
    source_files_cell = normalize_text(row.get("ソースのファイル名", ""))
    year = normalize_text(row.get("提出年度", ""))
    fmt = normalize_format(row.get("形式", ""))

    first_author = extract_first_author(author_text)
    old_names = split_filenames(source_files_cell)

    if not old_names:
        logs.append("SKIP: ソースのファイル名が空")
        return source_files_cell, logs

    if not title:
        logs.append("WARN: 研究タイトルが空のため、ファイル名中に untitled が入る可能性があります")
    if not year:
        logs.append("WARN: 提出年度が空のため、ファイル名の先頭が untitled になります")

    new_names: List[str] = []

    for i, old_name in enumerate(old_names, start=1):
        old_name = normalize_text(old_name)

        if looks_like_non_file_value(old_name):
            logs.append(f"SKIP_NON_FILE: {old_name}")
            new_names.append(old_name)
            continue

        old_path = find_archive_file(archive_dir, old_name)

        if old_path is None:
            logs.append(f"MISS: {archive_dir / old_name}")
            new_names.append(old_name)
            continue

        # すでに整っている名前ならそのまま（untitled を含む場合はメタ更新のため再計算する）
        if is_already_renamed(old_path.name) and not filename_has_untitled_placeholder(
            old_path.name
        ):
            logs.append(f"NO_RENAME_NEEDED: {old_path.name}")
            new_names.append(old_path.name)
            continue

        suffix = old_path.suffix or ".pdf"
        new_name = build_new_filename(
            year=year,
            fmt=fmt,
            first_author=first_author,
            title=title,
            index=i,
            suffix=suffix,
        )
        new_path = archive_dir / new_name

        if new_name == old_path.name:
            logs.append(f"NO_RENAME_NEEDED: {old_path.name}")
            new_names.append(old_path.name)
            continue

        # 同名で既に存在するなら、そのファイルが自分自身でなければ危険なのでスキップ
        if new_path.exists():
            if old_path.resolve() == new_path.resolve():
                logs.append(f"NO_RENAME_NEEDED: {old_path.name}")
                new_names.append(old_path.name)
            else:
                logs.append(f"SKIP_CONFLICT: {old_path.name} -> {new_path.name}")
                new_names.append(old_path.name)
            continue

        logs.append(f"RENAME: {old_path.name} -> {new_path.name}")

        if not dry_run:
            old_path.rename(new_path)

        new_names.append(new_path.name)

    new_cell_value = ", ".join(new_names)
    return new_cell_value, logs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="research_list.tsv を元に data/archives 内のファイル名を規則的に変更し、TSVへ反映する"
    )
    parser.add_argument("--tsv", type=Path, default=TSV_PATH)
    parser.add_argument("--archive-dir", type=Path, default=ARCHIVE_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-row", type=int, default=1)
    parser.add_argument("--end-row", type=int, default=0)
    args = parser.parse_args()

    tsv_path = args.tsv.resolve()
    archive_dir = args.archive_dir.resolve()

    if not tsv_path.exists():
        raise FileNotFoundError(f"TSVが見つかりません: {tsv_path}")
    if not archive_dir.exists():
        raise FileNotFoundError(f"archive-dir が見つかりません: {archive_dir}")

    rows, headers = read_tsv(tsv_path)
    validate_headers(headers)

    print(f"TSV: {tsv_path}")
    print(f"ARCHIVE_DIR: {archive_dir}")
    print(f"DRY_RUN: {args.dry_run}")
    print()

    changed_count = 0

    for idx, row in enumerate(rows, start=1):
        if idx < args.start_row:
            continue
        if args.end_row and idx > args.end_row:
            continue

        title = normalize_text(row.get("研究タイトル", ""))
        print(f"--- row {idx}: {title}")

        try:
            old_value = normalize_text(row.get("ソースのファイル名", ""))
            new_value, logs = rename_files_for_row(
                row=row,
                archive_dir=archive_dir,
                dry_run=args.dry_run,
            )

            for log in logs:
                print(log)

            if new_value != old_value:
                row["ソースのファイル名"] = new_value
                changed_count += 1
                print(f"UPDATE_TSV: {old_value} -> {new_value}")
            else:
                print("NO_CHANGE")

        except Exception as e:
            print(f"ERROR: row {idx} で失敗: {e}")

        print()

    if args.dry_run:
        print("dry-run のため、ファイル変更・TSV更新は行っていません。")
        return

    backup_path = backup_tsv(tsv_path)
    write_tsv(tsv_path, rows, headers)

    print(f"完了: {changed_count} 行を更新")
    print(f"TSVバックアップ: {backup_path}")
    print(f"更新後TSV: {tsv_path}")


if __name__ == "__main__":
    main()