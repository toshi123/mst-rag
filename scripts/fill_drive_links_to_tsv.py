#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Google Drive フォルダ内のファイルと research_list.tsv を照合し、
source_url / drive_file_id 列を追加・更新するスクリプト。

前提:
- TSVはタブ区切り
- 「ソースのファイル名」列を使ってGoogle Drive上のファイル名と照合する
- 複数ファイルは ", " または改行で区切られている
- 出力時は source_url / drive_file_id も ", " 区切りで保存する

使い方例:
python scripts/fill_drive_links_to_tsv.py \
  --tsv data/research_list.tsv \
  --folder-url "https://drive.google.com/drive/u/1/folders/1cMildJtLt9T5XhSogu6wAAaPGYOYy3E2" \
  --credentials /path/to/service_account.json \
  --out data/research_list_with_drive.tsv

上書きしたい場合:
python scripts/fill_drive_links_to_tsv.py \
  --tsv data/research_list.tsv \
  --folder-url "https://drive.google.com/drive/u/1/folders/1cMildJtLt9T5XhSogu6wAAaPGYOYy3E2" \
  --credentials /path/to/service_account.json \
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
SOURCE_FILENAME_COL = "ソースのファイル名"
SOURCE_URL_COL = "source_url"
DRIVE_FILE_ID_COL = "drive_file_id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TSVのファイル名とGoogle Drive内のファイルを照合して source_url / drive_file_id を埋める"
    )
    parser.add_argument("--tsv", required=True, help="入力TSVファイル")
    parser.add_argument("--folder-url", required=True, help="Google Drive フォルダURL")
    parser.add_argument(
        "--credentials",
        required=True,
        help="Google service account JSON のパス"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="出力TSVパス。省略時は入力TSVを上書き（--overwrite不要なら安全のため非推奨）"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存の source_url / drive_file_id がある行も上書きする"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細ログを出す"
    )
    return parser.parse_args()


def extract_folder_id(folder_url: str) -> str:
    """
    Google Drive フォルダURLからフォルダIDを抽出する。
    例:
    https://drive.google.com/drive/u/1/folders/1cMildJtLt9T5XhSogu6wAAaPGYOYy3E2
    """
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", folder_url)
    if not m:
        raise ValueError(f"フォルダIDをURLから抽出できませんでした: {folder_url}")
    return m.group(1)


def normalize_for_match(text: str) -> str:
    """
    ファイル名照合用のゆるい正規化。
    - NFKC正規化
    - 全角アンダースコア→半角
    - 改行/タブ/連続空白の正規化
    - 前後空白除去
    """
    if text is None:
        return ""
    s = text.strip()
    s = s.replace("＿", "_")
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def split_filenames(value: str) -> List[str]:
    """
    複数ファイル名を分割する。
    ルール:
    - ", " または改行区切り
    - "、" は区切りとして扱わない
    """
    if not value:
        return []

    value = value.strip()
    if not value:
        return []

    parts: List[str] = []
    for line in value.splitlines():
        line = line.strip()
        if not line:
            continue
        if ", " in line:
            parts.extend([x.strip() for x in line.split(", ") if x.strip()])
        else:
            parts.append(line)

    return parts


def build_drive_service(credentials_path: str):
    creds = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=SCOPES,
    )
    return build("drive", "v3", credentials=creds)


def list_files_in_folder(service, folder_id: str, verbose: bool = False) -> List[dict]:
    """
    指定フォルダ直下のファイルを列挙する。
    必要に応じて shared drives にも対応しやすいよう includeItemsFromAllDrives を付ける。
    """
    files: List[dict] = []
    page_token = None

    while True:
        response = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, mimeType, webViewLink)",
                pageSize=1000,
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )

        batch = response.get("files", [])
        files.extend(batch)

        if verbose:
            print(f"[INFO] fetched {len(batch)} files (total={len(files)})", file=sys.stderr)

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return files


def make_file_index(drive_files: List[dict]) -> Dict[str, List[dict]]:
    """
    正規化ファイル名 -> drive file 候補一覧
    """
    index: Dict[str, List[dict]] = {}
    for f in drive_files:
        key = normalize_for_match(f["name"])
        index.setdefault(key, []).append(f)
    return index


def ensure_columns(fieldnames: List[str]) -> List[str]:
    new_fields = list(fieldnames)

    if SOURCE_URL_COL not in new_fields:
        new_fields.append(SOURCE_URL_COL)

    if DRIVE_FILE_ID_COL not in new_fields:
        new_fields.append(DRIVE_FILE_ID_COL)

    return new_fields


def join_multi(values: List[str]) -> str:
    return ", ".join(values)


def should_skip_row(row: dict, overwrite: bool) -> bool:
    if overwrite:
        return False

    url_val = (row.get(SOURCE_URL_COL) or "").strip()
    id_val = (row.get(DRIVE_FILE_ID_COL) or "").strip()

    return bool(url_val or id_val)


def main() -> None:
    args = parse_args()

    tsv_path = Path(args.tsv)
    if not tsv_path.exists():
        raise FileNotFoundError(f"TSVが見つかりません: {tsv_path}")

    out_path = Path(args.out) if args.out else tsv_path
    folder_id = extract_folder_id(args.folder_url)

    if args.verbose:
        print(f"[INFO] folder_id = {folder_id}", file=sys.stderr)

    service = build_drive_service(args.credentials)
    drive_files = list_files_in_folder(service, folder_id, verbose=args.verbose)
    drive_index = make_file_index(drive_files)

    if args.verbose:
        print(f"[INFO] indexed drive files = {len(drive_files)}", file=sys.stderr)

    rows: List[dict] = []

    with tsv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("TSVヘッダが読めませんでした")

        fieldnames = ensure_columns(reader.fieldnames)

        if SOURCE_FILENAME_COL not in fieldnames:
            raise ValueError(f"必須列が見つかりません: {SOURCE_FILENAME_COL}")

        stats = {
            "rows_total": 0,
            "rows_updated": 0,
            "rows_skipped_existing": 0,
            "rows_no_source_filename": 0,
            "files_matched": 0,
            "files_missed": 0,
            "files_ambiguous": 0,
        }

        for row_idx, row in enumerate(reader, start=2):  # header = 1行目
            stats["rows_total"] += 1

            # 欠損列を補う
            for col in fieldnames:
                row.setdefault(col, "")

            if should_skip_row(row, overwrite=args.overwrite):
                stats["rows_skipped_existing"] += 1
                rows.append(row)
                continue

            src_value = (row.get(SOURCE_FILENAME_COL) or "").strip()
            if not src_value:
                stats["rows_no_source_filename"] += 1
                rows.append(row)
                continue

            filenames = split_filenames(src_value)
            matched_urls: List[str] = []
            matched_ids: List[str] = []

            if args.verbose:
                print(f"\n[ROW {row_idx}] source filenames = {filenames}", file=sys.stderr)

            for original_name in filenames:
                key = normalize_for_match(original_name)
                candidates = drive_index.get(key, [])

                if not candidates:
                    stats["files_missed"] += 1
                    print(
                        f"[MISS] row={row_idx} file='{original_name}'",
                        file=sys.stderr
                    )
                    continue

                if len(candidates) > 1:
                    stats["files_ambiguous"] += 1
                    print(
                        f"[AMBIGUOUS] row={row_idx} file='{original_name}' "
                        f"matches={[c['name'] for c in candidates]} -> first one is used",
                        file=sys.stderr
                    )

                chosen = candidates[0]
                file_id = chosen["id"]
                url = chosen.get("webViewLink") or f"https://drive.google.com/file/d/{file_id}/view"

                matched_ids.append(file_id)
                matched_urls.append(url)
                stats["files_matched"] += 1

                if args.verbose:
                    print(
                        f"[MATCH] row={row_idx} '{original_name}' -> '{chosen['name']}' ({file_id})",
                        file=sys.stderr
                    )

            if matched_ids or matched_urls:
                row[DRIVE_FILE_ID_COL] = join_multi(matched_ids)
                row[SOURCE_URL_COL] = join_multi(matched_urls)
                stats["rows_updated"] += 1

            rows.append(row)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter="\t",
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== DONE ===", file=sys.stderr)
    print(f"input_tsv:  {tsv_path}", file=sys.stderr)
    print(f"output_tsv: {out_path}", file=sys.stderr)
    print(f"rows_total: {stats['rows_total']}", file=sys.stderr)
    print(f"rows_updated: {stats['rows_updated']}", file=sys.stderr)
    print(f"rows_skipped_existing: {stats['rows_skipped_existing']}", file=sys.stderr)
    print(f"rows_no_source_filename: {stats['rows_no_source_filename']}", file=sys.stderr)
    print(f"files_matched: {stats['files_matched']}", file=sys.stderr)
    print(f"files_missed: {stats['files_missed']}", file=sys.stderr)
    print(f"files_ambiguous: {stats['files_ambiguous']}", file=sys.stderr)


if __name__ == "__main__":
    main()