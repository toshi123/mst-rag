#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import requests


CHROMA_DIR = Path("data/chroma")
DEFAULT_COLLECTIONS = ["mst_research", "esa_posts"]

OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "embeddinggemma"


def sanitize_filter_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip()
    return v if v else None


def normalize_source(source: Optional[str]) -> str:
    s = sanitize_filter_value(source)
    if s is None:
        return "all"
    s = s.lower()
    if s not in {"all", "pdf", "esa"}:
        raise ValueError(f"Invalid source: {source}. Use one of: all, pdf, esa")
    return s


def resolve_collections_for_source(source: str) -> List[str]:
    source = normalize_source(source)
    if source == "all":
        return ["mst_research", "esa_posts"]
    if source == "pdf":
        return ["mst_research"]
    if source == "esa":
        return ["esa_posts"]
    return DEFAULT_COLLECTIONS


def build_where_filter(
    year: Optional[str] = None,
    fmt: Optional[str] = None,
    genre: Optional[str] = None,
    project_id: Optional[str] = None,
    source_type: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    conditions: List[Dict[str, Any]] = []

    year = sanitize_filter_value(year)
    fmt = sanitize_filter_value(fmt)
    genre = sanitize_filter_value(genre)
    project_id = sanitize_filter_value(project_id)
    source_type = sanitize_filter_value(source_type)

    if year is not None:
        conditions.append({"year": year})
    if fmt is not None:
        conditions.append({"format": fmt})
    if genre is not None:
        conditions.append({"genre": genre})
    if project_id is not None:
        conditions.append({"project_id": project_id})
    if source_type is not None:
        conditions.append({"source_type": source_type})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def build_where_filter_for_collection(
    collection_name: str,
    year: Optional[str] = None,
    fmt: Optional[str] = None,
    genre: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if collection_name == "mst_research":
        return build_where_filter(
            year=year,
            fmt=fmt,
            genre=genre,
            project_id=project_id,
        )

    if collection_name == "esa_posts":
        return None

    return None


def infer_source_type(collection_name: str) -> str:
    if collection_name == "mst_research":
        return "pdf"
    if collection_name == "esa_posts":
        return "esa"
    return "unknown"


def embed_query(
    text: str,
    ollama_base_url: str = OLLAMA_BASE_URL,
    embed_model: str = EMBED_MODEL,
) -> List[float]:
    errors = []

    try:
        response = requests.post(
            f"{ollama_base_url}/api/embed",
            json={"model": embed_model, "input": [text]},
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        return data["embeddings"][0]
    except Exception as e:
        errors.append(f"/api/embed failed: {e}")

    try:
        response = requests.post(
            f"{ollama_base_url}/api/embeddings",
            json={"model": embed_model, "prompt": text},
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        return data["embedding"]
    except Exception as e:
        errors.append(f"/api/embeddings failed: {e}")

    raise RuntimeError("Embedding API failed on both endpoints:\n" + "\n".join(errors))


def get_client(chroma_dir: Path = CHROMA_DIR):
    return chromadb.PersistentClient(path=str(chroma_dir))


def get_collection(
    collection_name: str,
    chroma_dir: Path = CHROMA_DIR,
):
    client = get_client(chroma_dir=chroma_dir)
    return client.get_collection(collection_name)


def collection_exists(
    collection_name: str,
    chroma_dir: Path = CHROMA_DIR,
) -> bool:
    try:
        client = get_client(chroma_dir=chroma_dir)
        names = [c.name for c in client.list_collections()]
        return collection_name in names
    except Exception:
        return False


def run_query_single_collection(
    collection,
    collection_name: str,
    query_embedding: List[float],
    n_results: int,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]
    ids = result.get("ids", [[]])[0] if "ids" in result else [None] * len(docs)

    hits: List[Dict[str, Any]] = []
    source_type = infer_source_type(collection_name)

    for i, (doc, meta, dist, _id) in enumerate(zip(docs, metas, dists, ids), start=1):
        metadata = dict(meta or {})
        metadata.setdefault("source_type", source_type)
        metadata.setdefault("collection_name", collection_name)

        hits.append(
            {
                "rank_raw": i,
                "id": _id,
                "document": doc,
                "metadata": metadata,
                "distance": dist,
                "collection_name": collection_name,
                "source_type": source_type,
            }
        )
    return hits


def numeric_doc_priority(meta: Dict[str, Any]) -> int:
    raw = meta.get("doc_priority", 0)
    try:
        return int(raw)
    except Exception:
        return 0


def rerank_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(hit: Dict[str, Any]):
        dist = hit["distance"]
        meta = hit["metadata"]
        priority = numeric_doc_priority(meta)
        return (round(float(dist), 3), -priority)

    return sorted(hits, key=sort_key)


def apply_post_filters(
    hits: List[Dict[str, Any]],
    top_k: int,
    distance_cutoff: Optional[float] = None,
    max_per_doc: int = 2,
    max_per_project: int = 3,
    max_per_url: int = 2,
) -> List[Dict[str, Any]]:
    accepted: List[Dict[str, Any]] = []

    doc_counts = defaultdict(int)
    project_counts = defaultdict(int)
    url_counts = defaultdict(int)
    seen_dedupe_keys = set()

    for hit in hits:
        meta = hit["metadata"]
        dist = hit["distance"]
        source_type = hit.get("source_type") or meta.get("source_type") or "unknown"

        if distance_cutoff is not None and dist is not None and dist > distance_cutoff:
            continue

        if source_type == "pdf":
            dedupe_key = (meta.get("dedupe_key") or "").strip()
            if dedupe_key and dedupe_key in seen_dedupe_keys:
                continue

            doc_id = (meta.get("doc_id") or "").strip()
            if doc_id and doc_counts[doc_id] >= max_per_doc:
                continue

            proj_id = (meta.get("project_id") or "").strip()
            if proj_id and project_counts[proj_id] >= max_per_project:
                continue

            accepted.append(hit)

            if dedupe_key:
                seen_dedupe_keys.add(dedupe_key)
            if doc_id:
                doc_counts[doc_id] += 1
            if proj_id:
                project_counts[proj_id] += 1

        elif source_type == "esa":
            url = (meta.get("url") or meta.get("source_url") or "").strip()
            if url and url_counts[url] >= max_per_url:
                continue

            accepted.append(hit)

            if url:
                url_counts[url] += 1

        else:
            accepted.append(hit)

        if len(accepted) >= top_k:
            break

    return accepted


def retrieve_hits(
    query: str,
    top_k: int = 8,
    initial_k: int = 24,
    distance_cutoff: Optional[float] = None,
    max_per_doc: int = 2,
    max_per_project: int = 3,
    max_per_url: int = 2,
    year: Optional[str] = None,
    fmt: Optional[str] = None,
    genre: Optional[str] = None,
    project_id: Optional[str] = None,
    source: str = "all",
    collections: Optional[List[str]] = None,
    chroma_dir: Path = CHROMA_DIR,
    ollama_base_url: str = OLLAMA_BASE_URL,
    embed_model: str = EMBED_MODEL,
) -> Dict[str, Any]:
    target_collections = collections or resolve_collections_for_source(source)

    query_embedding = embed_query(
        text=query,
        ollama_base_url=ollama_base_url,
        embed_model=embed_model,
    )

    raw_hits_all: List[Dict[str, Any]] = []
    per_collection_filters: Dict[str, Any] = {}
    searched_collections: List[str] = []
    missing_collections: List[str] = []

    for collection_name in target_collections:
        if not collection_exists(collection_name, chroma_dir=chroma_dir):
            missing_collections.append(collection_name)
            continue

        where = build_where_filter_for_collection(
            collection_name=collection_name,
            year=year,
            fmt=fmt,
            genre=genre,
            project_id=project_id,
        )
        per_collection_filters[collection_name] = where

        collection = get_collection(
            collection_name=collection_name,
            chroma_dir=chroma_dir,
        )

        hits = run_query_single_collection(
            collection=collection,
            collection_name=collection_name,
            query_embedding=query_embedding,
            n_results=max(initial_k, top_k),
            where=where,
        )
        raw_hits_all.extend(hits)
        searched_collections.append(collection_name)

    reranked_hits = rerank_hits(raw_hits_all)

    final_hits = apply_post_filters(
        hits=reranked_hits,
        top_k=top_k,
        distance_cutoff=distance_cutoff,
        max_per_doc=max_per_doc,
        max_per_project=max_per_project,
        max_per_url=max_per_url,
    )

    return {
        "query": query,
        "filter": per_collection_filters,
        "raw_hits": raw_hits_all,
        "final_hits": final_hits,
        "collections": target_collections,
        "searched_collections": searched_collections,
        "missing_collections": missing_collections,
        "source": normalize_source(source),
    }


def unique_source_entries(hits: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    entries = []
    seen = set()

    for hit in hits:
        meta = hit["metadata"]
        source_type = hit.get("source_type") or meta.get("source_type") or "unknown"

        if source_type == "pdf":
            source_url = (meta.get("source_url") or "").strip()
            title = (meta.get("title") or "").strip()
            file_name = (meta.get("file_name") or "").strip()
            project_id = (meta.get("project_id") or "").strip()
            fmt = (meta.get("format") or "").strip()
            year = str(meta.get("year") or "").strip()

            key = ("pdf", source_url, title, file_name, project_id, fmt, year)
            if key in seen:
                continue
            seen.add(key)

            entries.append(
                {
                    "source_type": "pdf",
                    "title": title,
                    "file_name": file_name,
                    "project_id": project_id,
                    "format": fmt,
                    "year": year,
                    "source_url": source_url,
                }
            )

        elif source_type == "esa":
            url = (meta.get("url") or meta.get("source_url") or "").strip()
            title = (meta.get("title") or "").strip()
            category = (meta.get("category") or "").strip()
            heading = (meta.get("heading") or meta.get("section_title") or "").strip()

            key = ("esa", url, title, category, heading)
            if key in seen:
                continue
            seen.add(key)

            entries.append(
                {
                    "source_type": "esa",
                    "title": title,
                    "category": category,
                    "heading": heading,
                    "source_url": url,
                }
            )

        else:
            source_url = (meta.get("source_url") or "").strip()
            title = (meta.get("title") or "").strip()
            key = ("unknown", source_url, title)
            if key in seen:
                continue
            seen.add(key)

            entries.append(
                {
                    "source_type": "unknown",
                    "title": title,
                    "source_url": source_url,
                }
            )

    return entries


def build_context(final_hits: List[Dict[str, Any]]) -> str:
    blocks = []

    for i, hit in enumerate(final_hits, start=1):
        meta = hit["metadata"]
        source_type = hit.get("source_type") or meta.get("source_type") or "unknown"

        if source_type == "pdf":
            block = [
                f"[Source {i}]",
                "source_type: pdf",
                f"title: {meta.get('title', '')}",
                f"authors: {meta.get('authors', '')}",
                f"year: {meta.get('year', '')}",
                f"format: {meta.get('format', '')}",
                f"genre: {meta.get('genre', '')}",
                f"section_title: {meta.get('section_title', '')}",
                f"project_id: {meta.get('project_id', '')}",
                f"doc_id: {meta.get('doc_id', '')}",
                f"source_url: {meta.get('source_url', '')}",
                f"distance: {hit.get('distance', '')}",
                "content:",
                hit["document"].strip(),
            ]
        elif source_type == "esa":
            block = [
                f"[Source {i}]",
                "source_type: esa",
                f"title: {meta.get('title', '')}",
                f"category: {meta.get('category', '')}",
                f"tags: {meta.get('tags', '')}",
                f"heading: {meta.get('heading', '') or meta.get('section_title', '')}",
                f"url: {meta.get('url', '') or meta.get('source_url', '')}",
                f"distance: {hit.get('distance', '')}",
                "content:",
                hit["document"].strip(),
            ]
        else:
            block = [
                f"[Source {i}]",
                f"source_type: {source_type}",
                f"title: {meta.get('title', '')}",
                f"distance: {hit.get('distance', '')}",
                "content:",
                hit["document"].strip(),
            ]

        blocks.append("\n".join(block))

    return "\n\n" + "\n\n".join(blocks)


def shorten(text: str, n: int = 220) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= n:
        return text
    return text[: n - 1].rstrip() + "…"