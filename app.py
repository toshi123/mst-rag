from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Tuple

import chromadb
import requests
import streamlit as st


# =========================
# Config
# =========================

DEFAULT_CHROMA_DIR = os.environ.get(
    "CHROMA_DIR",
    "/Users/tsuji/Library/CloudStorage/Dropbox/Lecture/MSTC/develops/mst-rag/chroma_db",
)
DEFAULT_COLLECTION = os.environ.get("CHROMA_COLLECTION", "esa_posts")
DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "embeddinggemma:latest")
DEFAULT_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "qwen3:latest")


# =========================
# Ollama helpers
# =========================

def get_json(url: str, timeout: int = 60) -> Dict[str, Any]:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def post_json(url: str, payload: Dict[str, Any], timeout: int = 300) -> requests.Response:
    return requests.post(url, json=payload, timeout=timeout)


def list_models(host: str) -> List[str]:
    data = get_json(f"{host.rstrip('/')}/api/tags")
    models = data.get("models", [])
    return [m["name"] for m in models if isinstance(m.get("name"), str)]


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
                body = e.response.text[:300]
            except Exception:
                pass
            errors.append(f"{name}: HTTP {e.response.status_code} {body}")
        except Exception as e:
            errors.append(f"{name}: {e}")

    raise RuntimeError("All generation endpoints failed.\n" + "\n".join(errors))


# =========================
# Chroma helpers
# =========================

def unwrap_first(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        if not x:
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


def build_sources(rows: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    seen = set()
    sources: List[Tuple[str, str, str]] = []

    for row in rows:
        meta = row["metadata"]
        title = str(meta.get("title", "") or "")
        heading = str(meta.get("heading", "") or "")
        url = str(meta.get("url", "") or "")

        if not url or url in seen:
            continue
        seen.add(url)
        sources.append((title, heading, url))

    return sources


def format_source_label(title: str, heading: str) -> str:
    label = title.strip() if title else "(untitled)"
    if heading and heading != "INTRO":
        label += f" / {heading}"
    return label


def shorten(text: str, width: int = 160) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


# =========================
# Cached clients
# =========================

@st.cache_resource
def get_chroma_collection(chroma_dir: str, collection_name: str):
    client = chromadb.PersistentClient(path=chroma_dir)
    return client.get_collection(collection_name)


# =========================
# UI
# =========================

st.set_page_config(page_title="mst-rag", layout="wide")
st.title("mst-rag")
st.caption("esa を知識源にしたローカル RAG")

with st.sidebar:
    st.header("設定")
    chroma_dir = st.text_input("Chroma directory", value=DEFAULT_CHROMA_DIR)
    collection_name = st.text_input("Collection", value=DEFAULT_COLLECTION)
    ollama_host = st.text_input("Ollama host", value=DEFAULT_OLLAMA_HOST)
    embed_model = st.text_input("Embedding model", value=DEFAULT_EMBED_MODEL)
    llm_model = st.text_input("Chat model", value=DEFAULT_CHAT_MODEL)
    top_k = st.slider("Top K", min_value=1, max_value=12, value=5)
    max_per_url = st.slider("Max chunks per URL", min_value=1, max_value=4, value=1)
    show_retrieved = st.checkbox("検索候補を表示", value=True)

    try:
        models = list_models(ollama_host)
        st.caption("Ollama models")
        st.write(models)
    except Exception as e:
        st.warning(f"Ollama model list を取得できませんでした: {e}")

question = st.text_area(
    "質問",
    placeholder="例: MSTの進捗報告会はどのような形式ですか？",
    height=120,
)

run = st.button("検索して回答")

if run:
    if not question.strip():
        st.warning("質問を入力してください。")
        st.stop()

    try:
        collection = get_chroma_collection(chroma_dir, collection_name)
    except Exception as e:
        st.error(f"Chroma collection を開けませんでした: {e}")
        st.stop()

    try:
        count = collection.count()
    except Exception as e:
        st.error(f"Chroma count 取得に失敗しました: {e}")
        st.stop()

    st.caption(f"Collection count: {count}")

    if count == 0:
        st.error("コレクションが空です。先に sync_esa_to_chroma.py を実行してください。")
        st.stop()

    try:
        q_emb = embed_query(question, embed_model, ollama_host)
        result = collection.query(
            query_embeddings=[q_emb],
            n_results=top_k * max(2, max_per_url),
            include=["documents", "metadatas", "distances"],
        )
        rows = normalize_retrieval_result(result)
        rows = deduplicate_rows_by_url(rows, max_per_url=max_per_url)
        rows = rows[:top_k]
        context_docs = build_context_docs(rows)

        if not context_docs:
            st.error("使えるコンテキストを作れませんでした。")
            st.stop()

        answer = chat(question, context_docs, llm_model, ollama_host)
        sources = build_sources(rows)

    except Exception as e:
        st.error(f"検索または回答生成でエラーが発生しました: {e}")
        st.stop()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("回答")
        st.write(answer)

    with col2:
        st.subheader("参照元")
        if not sources:
            st.write("参照元なし")
        else:
            for i, (title, heading, url) in enumerate(sources, start=1):
                label = format_source_label(title, heading)
                st.markdown(f"**[{i}] {label}**")
                st.markdown(url)

    if show_retrieved:
        with st.expander("検索候補", expanded=False):
            if not rows:
                st.write("候補なし")
            else:
                for i, row in enumerate(rows, start=1):
                    meta = row["metadata"]
                    title = str(meta.get("title", "") or "")
                    heading = str(meta.get("heading", "") or "")
                    category = str(meta.get("category", "") or "")
                    url = str(meta.get("url", "") or "")
                    dist = row["distance"]
                    preview = shorten(row["document"], 220)

                    st.markdown(f"### [{i}] {format_source_label(title, heading)}")
                    st.write(f"distance: {dist}")
                    if category:
                        st.write(f"category: {category}")
                    if url:
                        st.markdown(url)
                    if preview:
                        st.write(preview)
                    st.divider()