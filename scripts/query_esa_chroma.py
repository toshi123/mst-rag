#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import chromadb
import requests


def post_json(url: str, payload: Dict[str, Any], timeout: int = 300) -> requests.Response:
    return requests.post(url, json=payload, timeout=timeout)


def get_json(url: str, timeout: int = 60) -> Dict[str, Any]:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def list_models(host: str) -> List[str]:
    data = get_json(f"{host.rstrip('/')}/api/tags")
    models = data.get("models", [])
    names: List[str] = []
    for m in models:
        name = m.get("name")
        if isinstance(name, str):
            names.append(name)
    return names


def embed_query(text: str, model: str, host: str) -> List[float]:
    url = f"{host.rstrip('/')}/api/embed"
    r = requests.post(url, json={"model": model, "input": text}, timeout=120)
    r.raise_for_status()
    data = r.json()

    if "embeddings" in data and isinstance(data["embeddings"], list):
        if not data["embeddings"]:
            raise RuntimeError("Embedding response contained empty 'embeddings'")
        return data["embeddings"][0]

    if "embedding" in data and isinstance(data["embedding"], list):
        return data["embedding"]

    raise RuntimeError(f"Unexpected embed response: {json.dumps(data, ensure_ascii=False)}")


def chat_via_api_chat(question: str, context_docs: List[str], model: str, host: str) -> str:
    url = f"{host.rstrip('/')}/api/chat"
    system_prompt = (
        "あなたはRAGアシスタントです。"
        "与えられたコンテキストに基づいてのみ回答してください。"
        "不明な場合は、わからないと答えてください。"
        "コンテキストにない内容を断定しないでください。"
        "回答は簡潔だが十分に具体的にしてください。"
    )
    context = "\n\n---\n\n".join(context_docs)

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"[コンテキスト]\n{context}\n\n[質問]\n{question}",
            },
        ],
    }

    r = post_json(url, payload, timeout=300)
    if r.status_code == 404:
        raise FileNotFoundError("/api/chat not found")
    r.raise_for_status()

    data = r.json()
    message = data.get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content

    raise RuntimeError(f"Unexpected /api/chat response: {json.dumps(data, ensure_ascii=False)}")


def chat_via_api_generate(question: str, context_docs: List[str], model: str, host: str) -> str:
    url = f"{host.rstrip('/')}/api/generate"
    context = "\n\n---\n\n".join(context_docs)
    prompt = f"""以下のコンテキストに基づいて質問に答えてください。
わからない場合は、わからないと述べてください。
コンテキストにないことを断定しないでください。
回答は簡潔だが十分に具体的にしてください。

[コンテキスト]
{context}

[質問]
{question}

[回答]
"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    r = post_json(url, payload, timeout=300)
    if r.status_code == 404:
        raise FileNotFoundError("/api/generate not found")
    r.raise_for_status()

    data = r.json()
    response = data.get("response")
    if isinstance(response, str):
        return response

    raise RuntimeError(f"Unexpected /api/generate response: {json.dumps(data, ensure_ascii=False)}")


def chat(question: str, docs: List[str], model: str, host: str) -> str:
    errors: List[str] = []

    for fn, name in [
        (chat_via_api_chat, "/api/chat"),
        (chat_via_api_generate, "/api/generate"),
    ]:
        try:
            return fn(question, docs, model, host)
        except FileNotFoundError as e:
            errors.append(f"{name}: not found ({e})")
        except requests.HTTPError as e:
            body = ""
            try:
                body = e.response.text[:500]
            except Exception:
                pass
            errors.append(f"{name}: HTTP {e.response.status_code} {body}")
        except Exception as e:
            errors.append(f"{name}: {e}")

    raise RuntimeError("All generation endpoints failed.\n" + "\n".join(errors))


def unwrap_first(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        if len(x) == 0:
            return []
        if isinstance(x[0], list):
            return x[0]
        return x
    return []


def normalize_retrieval_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_docs = unwrap_first(result.get("documents"))
    raw_metas = unwrap_first(result.get("metadatas"))
    raw_dists = unwrap_first(result.get("distances"))
    raw_ids = unwrap_first(result.get("ids"))

    n = max(len(raw_docs), len(raw_metas), len(raw_dists), len(raw_ids))
    rows: List[Dict[str, Any]] = []

    for i in range(n):
        doc = raw_docs[i] if i < len(raw_docs) else ""
        meta = raw_metas[i] if i < len(raw_metas) and isinstance(raw_metas[i], dict) else {}
        dist = raw_dists[i] if i < len(raw_dists) else None
        rid = raw_ids[i] if i < len(raw_ids) else None

        if doc is None:
            doc = ""
        if not isinstance(doc, str):
            doc = str(doc)

        rows.append(
            {
                "id": rid,
                "document": doc,
                "metadata": meta,
                "distance": dist,
            }
        )

    return rows


def shorten(text: str, width: int = 100) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def format_source_label(title: str, heading: str) -> str:
    label = title.strip() if title else "(untitled)"
    if heading and heading != "INTRO":
        label += f" / {heading}"
    return label


def build_context_docs(rows: List[Dict[str, Any]]) -> List[str]:
    context_docs: List[str] = []

    for row in rows:
        doc = row["document"].strip()
        meta = row["metadata"]

        if not doc:
            continue

        title = str(meta.get("title", "") or "")
        category = str(meta.get("category", "") or "")
        heading = str(meta.get("heading", "") or "")
        url = str(meta.get("url", "") or "")
        tags = str(meta.get("tags", "") or "")

        enriched = "\n".join(
            [
                f"title: {title}",
                f"category: {category}",
                f"heading: {heading}",
                f"tags: {tags}",
                f"url: {url}",
                "",
                doc,
            ]
        ).strip()

        context_docs.append(enriched)

    return context_docs


def deduplicate_rows_by_url(rows: List[Dict[str, Any]], max_per_url: int = 1) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    deduped: List[Dict[str, Any]] = []

    for row in rows:
        meta = row["metadata"]
        url = str(meta.get("url", "") or "")
        key = url if url else str(row["id"])

        current = counts.get(key, 0)
        if current >= max_per_url:
            continue

        counts[key] = current + 1
        deduped.append(row)

    return deduped


def build_sources(rows: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    seen = set()
    sources: List[Tuple[str, str, str]] = []

    for row in rows:
        meta = row["metadata"]
        title = str(meta.get("title", "") or "")
        heading = str(meta.get("heading", "") or "")
        url = str(meta.get("url", "") or "")

        if not url:
            continue
        if url in seen:
            continue

        seen.add(url)
        sources.append((title, heading, url))

    return sources


def print_question(question: str) -> None:
    print("質問:")
    print(f"  {question}")
    print()


def print_answer(answer: str) -> None:
    print("回答:")
    for line in answer.strip().splitlines():
        if line.strip():
            print(f"  {line}")
        else:
            print()
    print()


def print_sources(rows: List[Dict[str, Any]]) -> None:
    sources = build_sources(rows)

    print("参照元:")
    if not sources:
        print("  (なし)")
        print()
        return

    for i, (title, heading, url) in enumerate(sources, start=1):
        label = format_source_label(title, heading)
        print(f"  [{i}] {label}")
        print(f"      {url}")
        print()


def print_retrieved(rows: List[Dict[str, Any]], preview_width: int = 120) -> None:
    print("検索候補:")
    if not rows:
        print("  (なし)")
        print()
        return

    for i, row in enumerate(rows, start=1):
        meta = row["metadata"]
        doc = row["document"]
        dist = row["distance"]
        title = str(meta.get("title", "") or "")
        heading = str(meta.get("heading", "") or "")
        url = str(meta.get("url", "") or "")
        label = format_source_label(title, heading)
        preview = shorten(doc, preview_width)

        print(f"  [{i}] {label}")
        print(f"      dist: {dist}")
        if url:
            print(f"      {url}")
        if preview:
            print(f"      {preview}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--chroma-dir", default="./chroma_db")
    parser.add_argument("--collection", default="esa_posts")
    parser.add_argument("--embed-model", default=os.environ.get("OLLAMA_EMBED_MODEL", "embeddinggemma:latest"))
    parser.add_argument("--llm-model", default=os.environ.get("OLLAMA_CHAT_MODEL", "qwen3:latest"))
    parser.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--show-retrieved", action="store_true")
    parser.add_argument("--preview-width", type=int, default=120)
    parser.add_argument("--max-per-url", type=int, default=1, help="同じURLから採用する最大チャンク数")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.chroma_dir)
    collection = client.get_collection(args.collection)

    try:
        models = list_models(args.ollama_host)
    except Exception as e:
        raise RuntimeError(f"Could not query Ollama model list at {args.ollama_host}/api/tags: {e}")

    if args.llm_model not in models:
        print(f"Warning: llm model '{args.llm_model}' is not in Ollama model list.")

    if args.embed_model not in models:
        print(f"Warning: embed model '{args.embed_model}' is not in Ollama model list.")

    if collection.count() == 0:
        raise RuntimeError(
            f"Collection '{args.collection}' is empty. Run sync_esa_to_chroma.py first."
        )

    q_emb = embed_query(args.question, args.embed_model, args.ollama_host)

    result = collection.query(
        query_embeddings=[q_emb],
        n_results=args.k * max(2, args.max_per_url),
        include=["documents", "metadatas", "distances"],
    )

    rows = normalize_retrieval_result(result)
    rows = deduplicate_rows_by_url(rows, max_per_url=args.max_per_url)
    rows = rows[:args.k]

    context_docs = build_context_docs(rows)

    if not context_docs:
        raise RuntimeError("No usable context docs were built from Chroma results.")

    print_question(args.question)

    answer = chat(args.question, context_docs, args.llm_model, args.ollama_host)

    print_answer(answer)
    print_sources(rows)

    if args.show_retrieved:
        print_retrieved(rows, preview_width=args.preview_width)


if __name__ == "__main__":
    main()