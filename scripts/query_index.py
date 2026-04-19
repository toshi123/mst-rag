#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

from rag_retrieval import (
    retrieve_hits,
    unique_source_entries,
    shorten,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chroma から研究資料/esa を検索する"
    )
    parser.add_argument("query", help="検索クエリ")
    parser.add_argument("--source", choices=["all", "pdf", "esa"], default="all", help="検索対象ソース")
    parser.add_argument("--top-k", type=int, default=8, help="最終的に返す件数")
    parser.add_argument("--initial-k", type=int, default=24, help="後処理前に広めに取得する件数")
    parser.add_argument("--distance-cutoff", type=float, default=None, help="この距離より大きいものを除外する")
    parser.add_argument("--max-per-doc", type=int, default=2, help="同一 doc_id から採用する最大件数")
    parser.add_argument("--max-per-project", type=int, default=3, help="同一 project_id から採用する最大件数")
    parser.add_argument("--max-per-url", type=int, default=2, help="同一 esa URL から採用する最大件数")
    parser.add_argument("--year", default=None, help="metadata filter: 提出年度")
    parser.add_argument("--format", default=None, help="metadata filter: 形式 (Report / Poster / Title)")
    parser.add_argument("--genre", default=None, help="metadata filter: 研究分野")
    parser.add_argument("--project-id", default=None, help="metadata filter: project_id")
    parser.add_argument("--json", action="store_true", help="JSONで出力する")
    parser.add_argument("--show-doc", action="store_true", help="本文チャンクを表示する")
    parser.add_argument("--rerank", action="store_true", help="クロスエンコーダでリランク（sentence-transformers 必須）")
    parser.add_argument("--rerank-model", default=None, help="CrossEncoder のモデル名")
    return parser.parse_args()


def print_human_readable(
    query: str,
    source: str,
    where: Optional[Dict[str, Any]],
    raw_hits: List[Dict[str, Any]],
    final_hits: List[Dict[str, Any]],
    searched_collections: List[str],
    missing_collections: List[str],
    show_doc: bool = False,
) -> None:
    print(f"Query: {query}")
    print(f"Source: {source}")
    print(f"Filter: {json.dumps(where, ensure_ascii=False) if where else '(none)'}")
    print(f"Searched collections: {searched_collections}")
    if missing_collections:
        print(f"Missing collections: {missing_collections}")
    print(f"Raw hits: {len(raw_hits)}")
    print(f"Final hits: {len(final_hits)}")
    print("")

    for i, hit in enumerate(final_hits, start=1):
        meta = hit["metadata"]
        source_type = hit.get("source_type") or meta.get("source_type") or "unknown"

        print(f"[{i}] ({source_type}) {meta.get('title', '')}")
        print(f"  chroma_distance: {hit.get('chroma_distance', hit['distance'])}")
        if hit.get("rerank_score") is not None:
            print(f"  rerank_score  : {hit['rerank_score']}")
        print(f"  collection    : {hit.get('collection_name', '')}")
        print(f"  source_type   : {source_type}")

        if source_type == "pdf":
            print(f"  format/year   : {meta.get('format', '')} / {meta.get('year', '')}")
            print(f"  genre         : {meta.get('genre', '')}")
            print(f"  section       : {meta.get('section_title', '')}")
            print(f"  authors       : {meta.get('authors', '')}")
            print(f"  project_id    : {meta.get('project_id', '')}")
            print(f"  doc_id        : {meta.get('doc_id', '')}")
            print(f"  file_name     : {meta.get('file_name', '')}")
            print(f"  source_url    : {meta.get('source_url', '')}")
        elif source_type == "esa":
            print(f"  category      : {meta.get('category', '')}")
            print(f"  heading       : {meta.get('heading', '') or meta.get('section_title', '')}")
            print(f"  tags          : {meta.get('tags', '')}")
            print(f"  url           : {meta.get('url', '') or meta.get('source_url', '')}")
        else:
            print(f"  source_url    : {meta.get('source_url', '')}")

        if show_doc:
            print("  document:")
            print(f"    {hit['document'].replace(chr(10), chr(10) + '    ')}")
        else:
            print(f"  snippet       : {shorten(hit['document'])}")
        print("")

    refs = unique_source_entries(final_hits)
    if refs:
        print("References:")
        for ref in refs:
            source_type = ref.get("source_type", "unknown")
            if source_type == "pdf":
                label = ref["title"] or ref["file_name"] or ref["project_id"] or "(untitled)"
                print(f"  - (pdf) {label}")
                if ref.get("format") or ref.get("year"):
                    sub = " / ".join([x for x in [ref.get("format", ""), ref.get("year", "")] if x])
                    if sub:
                        print(f"    {sub}")
                if ref.get("source_url"):
                    print(f"    {ref['source_url']}")
            elif source_type == "esa":
                label = ref.get("title") or ref.get("category") or "(untitled)"
                print(f"  - (esa) {label}")
                if ref.get("category"):
                    print(f"    category: {ref['category']}")
                if ref.get("heading"):
                    print(f"    heading: {ref['heading']}")
                if ref.get("source_url"):
                    print(f"    {ref['source_url']}")
            else:
                label = ref.get("title") or "(untitled)"
                print(f"  - {label}")
                if ref.get("source_url"):
                    print(f"    {ref['source_url']}")
        print("")


def build_json_output(
    result: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "query": result["query"],
        "source": result["source"],
        "filter": result["filter"],
        "collections": result["collections"],
        "searched_collections": result["searched_collections"],
        "missing_collections": result["missing_collections"],
        "raw_hit_count": len(result["raw_hits"]),
        "final_hit_count": len(result["final_hits"]),
        "hits": result["final_hits"],
        "references": unique_source_entries(result["final_hits"]),
        "rerank_requested": result.get("rerank_requested"),
        "rerank_applied": result.get("rerank_applied"),
        "rerank_model": result.get("rerank_model"),
    }


def main() -> None:
    args = parse_args()

    result = retrieve_hits(
        query=args.query,
        source=args.source,
        top_k=args.top_k,
        initial_k=args.initial_k,
        distance_cutoff=args.distance_cutoff,
        max_per_doc=args.max_per_doc,
        max_per_project=args.max_per_project,
        max_per_url=args.max_per_url,
        year=args.year,
        fmt=args.format,
        genre=args.genre,
        project_id=args.project_id,
        use_cross_encoder_rerank=args.rerank,
        rerank_model=args.rerank_model,
    )

    if args.json:
        payload = build_json_output(result)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_human_readable(
            query=result["query"],
            source=result["source"],
            where=result["filter"],
            raw_hits=result["raw_hits"],
            final_hits=result["final_hits"],
            searched_collections=result["searched_collections"],
            missing_collections=result["missing_collections"],
            show_doc=args.show_doc,
        )


if __name__ == "__main__":
    main()