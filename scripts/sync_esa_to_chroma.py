#!/usr/bin/env python3
"""
esa MCP -> local sync -> Chroma registration
adapted for esa-mcp-server v0.7.x payload shape

既定の --chroma-dir は rag_retrieval.py / query_index.py と同じ data/chroma。
一覧 API では本文が空のことがあるため、必要時に esa_get_post で本文を取得する。

Requirements:
  pip install mcp chromadb requests tiktoken
  （チャンク分割は scripts/build_index.py と同じトークン基準・文境界・オーバーラップを利用するため、
   同じ環境の依存が必要です）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import chromadb
import requests
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from build_index import (
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
    OVERLAP_TOKENS,
    TARGET_CHUNK_TOKENS,
    chunk_paragraphs,
    split_paragraphs,
)


# ----------------------------
# Utilities
# ----------------------------

def iso_to_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def safe_json_loads(text: str) -> Any:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def first_nonempty(d: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in d and d[key] not in (None, "", [], {}):
            return d[key]
    return default


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def chunked(seq: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-") or "item"


def coerce_bool(value: Any, default: bool = False) -> bool:
    """esa API / MCP が wip を文字列で返すと bool(v) が誤る（例: bool('false') == True）。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "1", "yes", "y", "on"):
            return True
        if s in ("false", "0", "no", "n", "off", ""):
            return False
    return bool(value)


# ----------------------------
# MCP result normalization
# ----------------------------

def extract_text_blocks(result: Any) -> List[str]:
    texts: List[str] = []
    for block in getattr(result, "content", []) or []:
        if isinstance(block, types.TextContent):
            texts.append(block.text)
        else:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                texts.append(text)
    return texts


def normalize_call_result(result: Any) -> Any:
    structured = getattr(result, "structured_content", None)
    if structured not in (None, {}, []):
        return structured

    texts = extract_text_blocks(result)
    for text in texts:
        parsed = safe_json_loads(text)
        if parsed is not None:
            return parsed

    if texts:
        return texts[0]
    return None


def normalize_resource_result(resource_result: Any) -> Any:
    contents = getattr(resource_result, "contents", []) or []
    for content in contents:
        text = getattr(content, "text", None)
        if isinstance(text, str):
            parsed = safe_json_loads(text)
            if parsed is not None:
                return parsed
            return text
    return None


# ----------------------------
# Payload parsing for esa MCP actual shape
# ----------------------------

def parse_category_title_tags(cat_title_tags: str) -> Tuple[str, str, List[str]]:
    """
    Examples:
      "MST/2026/foo/bar #tag1 #tag2"
      "MST/2026/foo/bar"
      "title only #tag1"
    Heuristic:
      - strip trailing #tags
      - last path segment = title
      - preceding path = category
    """
    if not cat_title_tags:
        return "", "", []

    s = cat_title_tags.strip()

    tag_matches = re.findall(r"(?:^|\s)#([^\s#]+)", s)
    tags = [t.strip() for t in tag_matches if t.strip()]

    s_wo_tags = re.sub(r"(?:^|\s)#[^\s#]+", "", s).strip()

    if "/" in s_wo_tags:
        parts = [p.strip() for p in s_wo_tags.split("/") if p.strip()]
        if len(parts) >= 2:
            title = parts[-1]
            category = "/".join(parts[:-1])
            return category, title, tags

    return "", s_wo_tags, tags


def normalize_post_object(raw: Dict[str, Any]) -> Dict[str, Any]:
    ctt = first_nonempty(raw, ["category_and_title_and_tags"], "")
    parsed_category, parsed_title, parsed_tags = parse_category_title_tags(ctt)

    title = first_nonempty(raw, ["name", "title"], parsed_title)
    category = first_nonempty(raw, ["category", "category_path", "categoryPath"], parsed_category)
    tags = ensure_list(first_nonempty(raw, ["tags", "tag_names"], parsed_tags))

    updated_at = first_nonempty(raw, ["updated_at", "updatedAt"])
    created_at = first_nonempty(raw, ["created_at", "createdAt"])
    url = first_nonempty(raw, ["url", "html_url", "full_url"], "")
    body_md = first_nonempty(raw, ["body_md", "bodyMd", "body"], "")
    post_number = first_nonempty(raw, ["number", "post_number", "no"])
    post_id = first_nonempty(raw, ["id", "postId", "post_id"])
    wip = coerce_bool(first_nonempty(raw, ["wip", "is_wip"], False), False)

    return {
        "id": post_id,
        "number": post_number,
        "title": title,
        "category": category,
        "tags": [str(t) for t in tags],
        "updated_at": updated_at,
        "created_at": created_at,
        "url": url,
        "body_md": body_md,
        "wip": wip,
        "raw": raw,
    }


def extract_posts_from_payload(payload: Any) -> List[Dict[str, Any]]:
    if payload is None:
        return []

    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if isinstance(payload, dict):
        for key in ["posts", "items", "results", "data"]:
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]

        if any(k in payload for k in ["body_md", "updated_at", "url", "category_and_title_and_tags"]):
            return [payload]

    return []


def extract_post_number(post: Dict[str, Any]) -> Optional[int]:
    candidates = [
        post.get("number"),
        post.get("post_number"),
        post.get("no"),
    ]

    for value in candidates:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            pass

    url = str(post.get("url") or post.get("html_url") or post.get("full_url") or "")
    m = re.search(r"/posts/(\d+)", url)
    if m:
        return int(m.group(1))

    raw = post.get("raw")
    if isinstance(raw, dict):
        for key in ["number", "post_number", "no"]:
            value = raw.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                pass

        raw_url = str(raw.get("url") or raw.get("html_url") or raw.get("full_url") or "")
        m = re.search(r"/posts/(\d+)", raw_url)
        if m:
            return int(m.group(1))

    return None


def post_key(post: Dict[str, Any]) -> str:
    number = extract_post_number(post)
    if number is not None:
        return f"number:{number}"

    post_id = post.get("id")
    if post_id is not None:
        return f"id:{post_id}"

    return f"title:{post.get('title', 'unknown')}"


# ----------------------------
# Markdown chunking
# ----------------------------

@dataclass
class Chunk:
    heading: str
    text: str
    chunk_index: int


def chunk_markdown_by_heading(
    md: str,
    target_tokens: int = TARGET_CHUNK_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
    min_tokens: int = MIN_CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> List[Chunk]:
    """
    見出し単位でセクションを分け、各セクション本文は build_index.chunk_paragraphs と同じ
    （tiktoken によるトークン基準・句読点での文分割・チャンク間オーバーラップ）で分割する。
    """
    lines = md.splitlines()
    sections: List[Tuple[str, List[str]]] = []
    current_heading = "INTRO"
    current_lines: List[str] = []

    heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    for line in lines:
        m = heading_re.match(line)
        if m:
            if current_lines:
                sections.append((current_heading, current_lines))
            current_heading = m.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, current_lines))

    chunks: List[Chunk] = []
    chunk_index = 0
    for heading, sec_lines in sections:
        text = "\n".join(sec_lines).strip()
        if not text:
            continue
        paragraphs = split_paragraphs(text)
        for part in chunk_paragraphs(
            paragraphs,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            overlap_tokens=overlap_tokens,
        ):
            chunks.append(Chunk(heading=heading, text=part, chunk_index=chunk_index))
            chunk_index += 1

    return chunks


def make_embedding_input(post: Dict[str, Any], chunk: Chunk) -> str:
    tags = ", ".join(post["tags"])
    header = [
        f"title: {post['title']}",
        f"category: {post['category']}",
        f"heading: {chunk.heading}",
        f"tags: {tags}",
        f"updated_at: {post['updated_at']}",
        f"url: {post['url']}",
        "",
    ]
    return "\n".join(header) + chunk.text


# ----------------------------
# Ollama embedding
# ----------------------------

class OllamaEmbedder:
    def __init__(self, host: str, model: str, timeout: int = 120) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        url = f"{self.host}/api/embed"
        resp = requests.post(
            url,
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        if "embeddings" in data and isinstance(data["embeddings"], list):
            return data["embeddings"]

        if "embedding" in data and isinstance(data["embedding"], list):
            return [data["embedding"]]

        raise RuntimeError(f"Unexpected Ollama embed response: {data}")


# ----------------------------
# State
# ----------------------------

def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"last_synced_at": None, "posts": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------------------
# MCP client
# ----------------------------

ALIASES = {
    "team": ["teamName", "team_name", "team", "teamSlug", "team_slug"],
    "query": ["query", "q", "keyword", "search", "searchQuery", "search_query"],
    "page": ["page", "pageNum", "page_num"],
    "per_page": ["perPage", "per_page", "pageSize", "page_size", "limit", "count"],
    "post_number": ["postNumber", "post_number", "number", "no", "postNo", "post_no"],
}

def get_tool_schema(tool: Any) -> Dict[str, Any]:
    schema = getattr(tool, "inputSchema", None)
    return schema if isinstance(schema, dict) else {}

def schema_properties(schema: Dict[str, Any]) -> Dict[str, Any]:
    props = schema.get("properties", {})
    return props if isinstance(props, dict) else {}

def pick_schema_key(schema: Dict[str, Any], semantic_name: str) -> Optional[str]:
    props = schema_properties(schema)
    for alias in ALIASES.get(semantic_name, []):
        if alias in props:
            return alias
    return None

def build_search_args(schema: Dict[str, Any], team: str, query: str, page: int, per_page: int) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    for semantic_name, value in [
        ("team", team),
        ("query", query),
        ("page", page),
        ("per_page", per_page),
    ]:
        key = pick_schema_key(schema, semantic_name)
        if key is not None:
            args[key] = value
    return args


def build_get_post_args(schema: Dict[str, Any], team: str, post_number: int) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    team_key = pick_schema_key(schema, "team")
    if team_key is not None:
        args[team_key] = team
    num_key = pick_schema_key(schema, "post_number")
    if num_key is not None:
        props = schema_properties(schema)
        prop = props.get(num_key, {})
        # JSON Schema の type が integer なら int を渡す（一部サーバは厳密）
        if prop.get("type") == "integer":
            args[num_key] = int(post_number)
        else:
            args[num_key] = post_number
    return args


class EsaMCPClient:
    def __init__(self, command: str, args: List[str], env: Dict[str, str]) -> None:
        self.command = command
        self.args = args
        self.env = env
        self._session: Optional[ClientSession] = None
        self._stdio_cm = None
        self._session_cm = None
        self.tools: Dict[str, Any] = {}

    async def __aenter__(self) -> "EsaMCPClient":
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
        )

        self._stdio_cm = stdio_client(server_params)
        read, write = await self._stdio_cm.__aenter__()

        self._session_cm = ClientSession(read, write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()

        listed = await self._session.list_tools()
        self.tools = {tool.name: tool for tool in listed.tools}

        print("=== MCP Tools ===")
        for name in sorted(self.tools.keys()):
            print("-", name)
        print()

        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(exc_type, exc, tb)
            except Exception:
                pass

        if self._stdio_cm is not None:
            try:
                await self._stdio_cm.__aexit__(exc_type, exc, tb)
            except Exception:
                pass

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("Session is not initialized")
        return self._session

    async def read_recent_posts(self, team: str) -> List[Dict[str, Any]]:
        uri = f"esa://teams/{team}/posts/recent"
        result = await self.session.read_resource(uri)
        payload = normalize_resource_result(result)
        posts = extract_posts_from_payload(payload)
        normalized = [normalize_post_object(p) for p in posts]
        print(f"recent_posts found: {len(normalized)}")
        return normalized

    async def search_posts(
        self,
        team: str,
        query: str = "",
        max_pages: int = 20,
        per_page: int = 100,
    ) -> List[Dict[str, Any]]:
        tool = self.tools.get("esa_search_posts")
        if tool is None:
            print("esa_search_posts tool is not available")
            return []

        schema = get_tool_schema(tool)
        found: Dict[str, Dict[str, Any]] = {}

        for page in range(1, max_pages + 1):
            args = build_search_args(schema, team=team, query=query, page=page, per_page=per_page)
            print(f"search_posts page={page} args={args}")
            result = await self.session.call_tool("esa_search_posts", arguments=args)
            payload = normalize_call_result(result)
            posts = [normalize_post_object(p) for p in extract_posts_from_payload(payload)]
            print(f"search_posts page={page} -> {len(posts)} posts")

            if not posts:
                break

            for post in posts:
                found[post_key(post)] = post

            if len(posts) < per_page:
                break

        normalized = list(found.values())
        print(f"search_posts total unique: {len(normalized)}")
        return normalized

    async def get_post(self, team: str, post_number: int) -> Optional[Dict[str, Any]]:
        tool = self.tools.get("esa_get_post")
        if tool is None:
            print("esa_get_post tool is not available")
            return None

        schema = get_tool_schema(tool)
        args = build_get_post_args(schema, team=team, post_number=post_number)
        if not args:
            print("esa_get_post: could not map arguments from tool schema")
            return None

        result = await self.session.call_tool("esa_get_post", arguments=args)
        payload = normalize_call_result(result)
        posts = extract_posts_from_payload(payload)
        if posts:
            return normalize_post_object(posts[0])
        if isinstance(payload, dict) and any(
            k in payload for k in ["body_md", "bodyMd", "body", "name", "title", "number"]
        ):
            return normalize_post_object(payload)
        return None


# ----------------------------
# Main sync
# ----------------------------

async def collect_candidate_posts(client: EsaMCPClient, team: str, bootstrap: bool) -> List[Dict[str, Any]]:
    found: Dict[str, Dict[str, Any]] = {}

    if bootstrap:
        posts = await client.search_posts(team=team, query="")
        for p in posts:
            found[post_key(p)] = p

    recent = await client.read_recent_posts(team=team)
    for p in recent:
        found[post_key(p)] = p

    merged = list(found.values())
    print(f"merged candidates: {len(merged)}")
    return merged


def should_index_post(post: Dict[str, Any], state: Dict[str, Any], category_prefix: Optional[str]) -> bool:
    if category_prefix:
        category = post.get("category") or ""
        if not category.startswith(category_prefix):
            return False

    key = post_key(post)
    old = state["posts"].get(key)
    if not old:
        return True

    old_dt = iso_to_dt(old.get("updated_at"))
    new_dt = iso_to_dt(post.get("updated_at"))
    if old_dt is None or new_dt is None:
        return True

    return new_dt > old_dt


def build_chroma_records(team: str, post: Dict[str, Any], chunks: List[Chunk]) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    post_id_part = str(extract_post_number(post) or post.get("id") or slugify(post.get("title", "post")))

    for chunk in chunks:
        chunk_id = f"esa:{team}:{post_id_part}:{chunk.chunk_index}"
        ids.append(chunk_id)
        docs.append(make_embedding_input(post, chunk))
        pn = extract_post_number(post)
        metas.append(
            {
                "source": "esa",
                "team": team,
                "post_number": int(pn) if pn is not None else -1,
                "post_id": str(post.get("id") or ""),
                "title": post.get("title") or "",
                "category": post.get("category") or "",
                "tags": ", ".join(post.get("tags") or []),
                "updated_at": post.get("updated_at") or "",
                "created_at": post.get("created_at") or "",
                "url": post.get("url") or "",
                "heading": chunk.heading,
                "chunk_index": chunk.chunk_index,
                "wip": coerce_bool(post.get("wip"), False),
            }
        )

    return ids, docs, metas


async def run_sync(args: argparse.Namespace) -> None:
    if "ESA_ACCESS_TOKEN" not in os.environ:
        raise RuntimeError("ESA_ACCESS_TOKEN is not set")

    env = {
        "ESA_ACCESS_TOKEN": os.environ["ESA_ACCESS_TOKEN"],
        "LANG": os.environ.get("LANG", "ja"),
    }

    if args.mcp_mode == "docker":
        command = "docker"
        mcp_args = [
            "run",
            "-i",
            "--rm",
            "-e", "ESA_ACCESS_TOKEN",
            "-e", "LANG",
            "ghcr.io/esaio/esa-mcp-server",
        ]
    else:
        command = args.npx_command
        mcp_args = ["@esaio/esa-mcp-server"]

    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "embeddinggemma:latest")

    state_path = Path(args.state_file)
    chroma_dir = Path(args.chroma_dir)

    print("=== Paths ===")
    print("chroma_dir =", chroma_dir.resolve())
    print("state_file =", state_path.resolve())
    print()

    state = load_state(state_path)

    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = chroma_client.get_or_create_collection(args.collection)
    embedder = OllamaEmbedder(host=ollama_host, model=embed_model)

    print("=== Collection ===")
    print("collection =", args.collection)
    print("count_before =", collection.count())
    print()

    async with EsaMCPClient(command=command, args=mcp_args, env=env) as client:
        candidates = await collect_candidate_posts(client, team=args.team, bootstrap=args.bootstrap)

        if not candidates:
            print("No candidate posts found.")
            return

        targets = [
            p for p in candidates
            if should_index_post(p, state=state, category_prefix=args.category_prefix)
        ]

        print(f"Candidates: {len(candidates)}")
        print(f"To index : {len(targets)}")
        print()

        indexed_count = 0
        skipped_count = 0
        failed_count = 0

        for post in targets:
            try:
                title = post.get("title") or "(untitled)"
                post_number = extract_post_number(post)
                body = post.get("body_md") or ""

                print("=" * 80)
                print(f"Processing: {title}")
                print(f"  number={post_number}")
                print(f"  url={post.get('url')}")
                print(f"  category={post.get('category')}")
                print(f"  tags={post.get('tags')}")
                print(f"  body_length={len(body)}")

                if not body.strip() and post_number is not None:
                    print(f"  fetching full post via esa_get_post...")
                    fetched = await client.get_post(team=args.team, post_number=post_number)
                    if fetched:
                        body = fetched.get("body_md") or ""
                        post = fetched
                        print(f"  body_length_after_fetch={len(body)}")
                    else:
                        print("  esa_get_post returned no body")

                if args.skip_wip and coerce_bool(post.get("wip"), False):
                    print("  Skip WIP")
                    skipped_count += 1
                    continue

                if not body.strip():
                    print("  Skip empty body")
                    skipped_count += 1
                    continue

                chunks = chunk_markdown_by_heading(
                    body,
                    target_tokens=args.target_tokens,
                    max_tokens=args.max_tokens,
                    min_tokens=args.min_tokens,
                    overlap_tokens=args.overlap_tokens,
                )
                print(f"  chunks={len(chunks)}")

                if not chunks:
                    print("  Skip no chunks")
                    skipped_count += 1
                    continue

                ids, docs, metas = build_chroma_records(args.team, post, chunks)

                key = post_key(post)
                old_entry = state["posts"].get(key, {})
                old_ids = old_entry.get("chunk_ids", [])

                if old_ids:
                    try:
                        collection.delete(ids=old_ids)
                        print(f"  deleted old chunks={len(old_ids)}")
                    except Exception as exc:
                        print(f"  Warning: failed to delete old chunks: {exc}")

                embeddings: List[List[float]] = []
                for idx, batch in enumerate(chunked(docs, args.embed_batch_size), start=1):
                    print(f"  embedding batch {idx}: size={len(batch)}")
                    batch_emb = embedder.embed_texts(batch)
                    embeddings.extend(batch_emb)

                if not ids or not docs or not metas or not embeddings:
                    print("  Skip invalid records")
                    skipped_count += 1
                    continue

                if not (len(ids) == len(docs) == len(metas) == len(embeddings)):
                    raise RuntimeError(
                        f"Length mismatch ids={len(ids)} docs={len(docs)} metas={len(metas)} embeddings={len(embeddings)}"
                    )

                print(f"  upsert records={len(ids)}")
                collection.upsert(
                    ids=ids,
                    documents=docs,
                    metadatas=metas,
                    embeddings=embeddings,
                )

                state["posts"][key] = {
                    "updated_at": post.get("updated_at"),
                    "chunk_ids": ids,
                    "title": post.get("title"),
                    "url": post.get("url"),
                    "category": post.get("category"),
                }

                indexed_count += 1
                print(f"  Indexed: {title} ({len(ids)} chunks)")

            except Exception as exc:
                failed_count += 1
                print(f"  Skip failed post: {post.get('title')} -> {exc}")
                continue

        print()
        print("=== Summary ===")
        print("indexed_count =", indexed_count)
        print("skipped_count =", skipped_count)
        print("failed_count  =", failed_count)
        print()

    state["last_synced_at"] = datetime.now(timezone.utc).isoformat()
    save_state(state_path, state)

    final_count = collection.count()
    print("=== Final ===")
    print(f"Final collection count: {final_count}")
    print(f"Done. State saved to {state_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--team", required=True, help="esa team name")
    p.add_argument("--collection", default="esa_posts")
    p.add_argument(
        "--chroma-dir",
        default="data/chroma",
        help="rag_retrieval.py / query_index.py と同じ既定 (data/chroma) に揃えています",
    )
    p.add_argument("--state-file", default="./state/esa_sync_state.json")
    p.add_argument("--category-prefix", default=None)
    p.add_argument("--skip-wip", action="store_true")
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument(
        "--target-tokens",
        type=int,
        default=TARGET_CHUNK_TOKENS,
        help=f"チャンク目標トークン数（既定: {TARGET_CHUNK_TOKENS}、build_index と同じ）",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_CHUNK_TOKENS,
        help=f"チャンク上限トークン数（既定: {MAX_CHUNK_TOKENS}）",
    )
    p.add_argument(
        "--min-tokens",
        type=int,
        default=MIN_CHUNK_TOKENS,
        help=f"これ未満のチャンクは前と結合（既定: {MIN_CHUNK_TOKENS}）",
    )
    p.add_argument(
        "--overlap-tokens",
        type=int,
        default=OVERLAP_TOKENS,
        help=f"隣接チャンク間のオーバーラップ（既定: {OVERLAP_TOKENS}）",
    )
    p.add_argument("--embed-batch-size", type=int, default=32)
    p.add_argument("--mcp-mode", choices=["docker", "npx"], default="docker")
    p.add_argument("--npx-command", default="npx")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run_sync(args))


if __name__ == "__main__":
    main()