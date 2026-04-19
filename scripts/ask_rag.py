#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

import requests

from rag_retrieval import (
    retrieve_hits,
    unique_source_entries,
    build_context,
)


OLLAMA_BASE_URL = "http://localhost:11434"
GENERATE_MODEL = "qwen2.5:7b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="研究資料/esa を対象にRAG回答を生成する"
    )
    parser.add_argument("query", help="質問文")
    parser.add_argument("--source", choices=["all", "pdf", "esa"], default="all", help="検索対象ソース")
    parser.add_argument("--top-k", type=int, default=8, help="最終的に使うチャンク数")
    parser.add_argument("--initial-k", type=int, default=24, help="最初に広めに取る件数")
    parser.add_argument("--distance-cutoff", type=float, default=None, help="距離しきい値")
    parser.add_argument("--max-per-doc", type=int, default=2, help="同一 doc_id から採用する最大件数")
    parser.add_argument("--max-per-project", type=int, default=3, help="同一 project_id から採用する最大件数")
    parser.add_argument("--max-per-url", type=int, default=2, help="同一 esa URL から採用する最大件数")
    parser.add_argument("--year", default=None, help="metadata filter: 提出年度")
    parser.add_argument("--format", default=None, help="metadata filter: 形式")
    parser.add_argument("--genre", default=None, help="metadata filter: 研究分野")
    parser.add_argument("--project-id", default=None, help="metadata filter: project_id")
    parser.add_argument("--model", default=GENERATE_MODEL, help="生成モデル")
    parser.add_argument("--show-context", action="store_true", help="生成に使ったコンテキストも表示")
    parser.add_argument("--json", action="store_true", help="JSONで出力")
    return parser.parse_args()


def build_prompt(user_query: str, context: str) -> str:
    return f"""あなたは、学校内の研究資料アーカイブとesa記事アーカイブをもとに回答するアシスタントです。
以下のコンテキストだけを根拠として回答してください。
根拠が足りない場合は、足りないと明記してください。
推測で断定しないでください。

回答方針:
- まず質問に対する要点を簡潔に述べる
- 必要に応じて「共通点」「相違点」「研究の流れ」「今後の論点」で整理する
- 文脈上わかる範囲で、どの資料群に基づくかが伝わるように書く
- コンテキストにないことは言い切らない
- 最後に「参照資料:」という見出しは付けない（別で整形表示するため）

質問:
{user_query}

コンテキスト:
{context}
""".strip()


def generate_answer(prompt: str, model: str, ollama_base_url: str = OLLAMA_BASE_URL) -> str:
    response = requests.post(
        f"{ollama_base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=600,
    )
    response.raise_for_status()
    data = response.json()
    return data["response"].strip()


def format_references(refs: List[Dict[str, str]]) -> str:
    if not refs:
        return ""

    lines = ["参照資料:"]
    for ref in refs:
        source_type = ref.get("source_type", "unknown")

        if source_type == "pdf":
            label_parts = []
            if ref.get("title"):
                label_parts.append(ref["title"])
            if ref.get("format"):
                label_parts.append(ref["format"])
            if ref.get("year"):
                label_parts.append(ref["year"])

            label = " / ".join(label_parts) if label_parts else (ref.get("file_name") or ref.get("project_id") or "(untitled)")
            lines.append(f"- (pdf) {label}")
            if ref.get("source_url"):
                lines.append(f"  {ref['source_url']}")

        elif source_type == "esa":
            label_parts = []
            if ref.get("title"):
                label_parts.append(ref["title"])
            if ref.get("category"):
                label_parts.append(f"category={ref['category']}")
            if ref.get("heading"):
                label_parts.append(f"heading={ref['heading']}")

            label = " / ".join(label_parts) if label_parts else "(esa)"
            lines.append(f"- (esa) {label}")
            if ref.get("source_url"):
                lines.append(f"  {ref['source_url']}")

        else:
            label = ref.get("title") or "(untitled)"
            lines.append(f"- {label}")
            if ref.get("source_url"):
                lines.append(f"  {ref['source_url']}")

    return "\n".join(lines)


def build_json_output(
    result: Dict[str, Any],
    answer: str,
    refs: List[Dict[str, str]],
    context: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    payload = {
        "query": result["query"],
        "source": result["source"],
        "filter": result["filter"],
        "collections": result["collections"],
        "searched_collections": result["searched_collections"],
        "missing_collections": result["missing_collections"],
        "raw_hit_count": len(result["raw_hits"]),
        "final_hit_count": len(result["final_hits"]),
        "answer": answer,
        "hits": result["final_hits"],
        "references": refs,
    }
    if context is not None:
        payload["context"] = context
    if prompt is not None:
        payload["prompt"] = prompt
    return payload


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
    )

    context = build_context(result["final_hits"])
    prompt = build_prompt(result["query"], context)
    answer = generate_answer(prompt, model=args.model)

    refs = unique_source_entries(result["final_hits"])
    refs_text = format_references(refs)

    if args.json:
        payload = build_json_output(
            result=result,
            answer=answer,
            refs=refs,
            context=context if args.show_context else None,
            prompt=prompt if args.show_context else None,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if result["missing_collections"]:
        print(f"[warning] missing collections: {result['missing_collections']}\n")

    print(answer)
    print("")
    if refs_text:
        print(refs_text)

    if args.show_context:
        print("\n--- CONTEXT ---")
        print(context)


if __name__ == "__main__":
    main()