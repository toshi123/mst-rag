#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import functools
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import chromadb
import pymupdf4llm
import requests
import tiktoken
from tqdm import tqdm


# =========================
# 設定
# =========================

TSV_PATH = Path("data/research_list.tsv")
ARCHIVES_DIR = Path("data/archives")
CHROMA_DIR = Path("data/chroma")
MARKDOWN_CACHE_DIR = Path("data/processed_markdown")
COLLECTION_NAME = "mst_research"

OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "embeddinggemma"

SOURCE_FILENAME_COL = "ソースのファイル名"

# チャンク長は tiktoken（既定 cl100k_base）で数える。埋め込みモデルの実トークナイザとは一致しないが、
# 長さの目安として安定させるため使う。
ENCODING_NAME = "cl100k_base"

MIN_CHUNK_TOKENS = 128
TARGET_CHUNK_TOKENS = 400
MAX_CHUNK_TOKENS = 550
OVERLAP_TOKENS = 80

# 同一文書内の重複抑制
NEAR_DUPLICATE_SIMILARITY = 0.92
MAX_NEAR_DUPLICATE_COMPARISONS = 20  # 直近N個だけ比較して軽く抑制

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
MULTISPACE_RE = re.compile(r"[ \t]+")
MULTINEWLINE_RE = re.compile(r"\n{3,}")
NON_WORD_RE = re.compile(r"[^\wぁ-んァ-ヶ一-龠]+")

FORMAT_PRIORITY = {
    "Report": 3,
    "Poster": 2,
    "Title": 1,
}


# =========================
# データ構造
# =========================

@dataclass
class Section:
    heading_level: int
    heading_text: str
    body: str


@dataclass
class Chunk:
    section_title: str
    section_level: int
    chunk_index_in_section: int
    text: str


# =========================
# 汎用
# =========================

def split_filenames(value: str) -> List[str]:
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


def normalize_for_match(text: str) -> str:
    if text is None:
        return ""
    s = text.strip()
    s = s.replace("＿", "_")
    s = unicodedata.normalize("NFKC", s)
    s = MULTISPACE_RE.sub(" ", s)
    return s.strip()


def normalize_text_for_dedupe(text: str) -> str:
    """
    重複判定用の強め正規化。
    """
    s = unicodedata.normalize("NFKC", text)
    s = s.lower()
    s = s.replace("＿", "_")
    s = s.replace("\n", " ")
    s = MULTISPACE_RE.sub(" ", s)
    s = NON_WORD_RE.sub("", s)
    return s.strip()


def sanitize_metadata_value(value):
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def md5_short(text: str, n: int = 12) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]


def slugify_for_id(text: str) -> str:
    s = unicodedata.normalize("NFKC", text).strip().lower()
    s = s.replace("＿", "_")
    s = MULTISPACE_RE.sub(" ", s)
    s = NON_WORD_RE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def get_head_author(authors: str) -> str:
    if not authors:
        return ""
    tmp = authors.replace("、", ",").replace("，", ",")
    parts = [p.strip() for p in tmp.split(",") if p.strip()]
    return parts[0] if parts else authors.strip()


def make_project_id(title: str, authors: str, year: str) -> str:
    """
    同一研究を Poster / Report / Title で束ねるための研究単位ID。
    基本は 年度 + 先頭著者 + 研究タイトル。
    """
    head_author = get_head_author(authors)
    key = " | ".join([
        slugify_for_id(year or ""),
        slugify_for_id(head_author or ""),
        slugify_for_id(title or ""),
    ])
    return f"proj_{md5_short(key, 16)}"


def get_doc_priority(fmt: str) -> int:
    return FORMAT_PRIORITY.get((fmt or "").strip(), 0)


def load_rows(tsv_path: Path) -> List[Dict[str, str]]:
    with tsv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def build_archive_index(archives_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for p in archives_dir.iterdir():
        if p.is_file():
            index[normalize_for_match(p.name)] = p
    return index


def find_archive_file(filename: str, archive_index: Dict[str, Path]) -> Optional[Path]:
    p = ARCHIVES_DIR / filename
    if p.exists():
        return p
    return archive_index.get(normalize_for_match(filename))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# =========================
# PDF -> Markdown
# =========================

def extract_pdf_markdown(pdf_path: Path) -> str:
    return pymupdf4llm.to_markdown(str(pdf_path))


def markdown_cache_path(pdf_path: Path) -> Path:
    return MARKDOWN_CACHE_DIR / f"{pdf_path.stem}.md"


def get_markdown_for_pdf(pdf_path: Path, use_cache: bool = True) -> str:
    ensure_dir(MARKDOWN_CACHE_DIR)
    cache_path = markdown_cache_path(pdf_path)

    if use_cache and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    markdown_text = extract_pdf_markdown(pdf_path)
    cache_path.write_text(markdown_text, encoding="utf-8")
    return markdown_text


# =========================
# Markdown 整形
# =========================

def clean_markdown(md: str) -> str:
    lines = [line.rstrip() for line in md.splitlines()]
    text = "\n".join(lines)
    text = MULTINEWLINE_RE.sub("\n\n", text)
    return text.strip()


def ensure_fallback_heading(md: str, fallback_title: str) -> str:
    has_heading = any(HEADING_RE.match(line) for line in md.splitlines())
    if has_heading:
        return md

    heading = fallback_title.strip() if fallback_title.strip() else "Document"
    return f"# {heading}\n\n{md.strip()}"


# =========================
# Markdown -> Sections
# =========================

def parse_sections_from_markdown(md: str) -> List[Section]:
    lines = md.splitlines()

    sections: List[Section] = []
    current_heading_level = 1
    current_heading_text = "Document"
    current_body_lines: List[str] = []

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            body = "\n".join(current_body_lines).strip()
            if body:
                sections.append(
                    Section(
                        heading_level=current_heading_level,
                        heading_text=current_heading_text,
                        body=body,
                    )
                )
            current_heading_level = len(m.group(1))
            current_heading_text = m.group(2).strip()
            current_body_lines = []
        else:
            current_body_lines.append(line)

    body = "\n".join(current_body_lines).strip()
    if body:
        sections.append(
            Section(
                heading_level=current_heading_level,
                heading_text=current_heading_text,
                body=body,
            )
        )

    return sections


# =========================
# Section -> Chunks
# =========================

def split_paragraphs(text: str) -> List[str]:
    raw_parts = re.split(r"\n\s*\n", text.strip())
    paragraphs = [MULTISPACE_RE.sub(" ", p.replace("\n", " ")).strip() for p in raw_parts]
    return [p for p in paragraphs if p]


@functools.lru_cache(maxsize=1)
def get_token_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str, enc: tiktoken.Encoding) -> int:
    if not text:
        return 0
    return len(enc.encode(text))


def split_into_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[。．！？!?])\s+|(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def split_tokens_hard(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
    enc: tiktoken.Encoding,
) -> List[str]:
    ids = enc.encode(text)
    if len(ids) <= max_tokens:
        return [text]
    stride = max_tokens - overlap_tokens
    if stride <= 0:
        stride = max(1, max_tokens // 2)
    out: List[str] = []
    i = 0
    while i < len(ids):
        chunk_ids = ids[i : i + max_tokens]
        out.append(enc.decode(chunk_ids))
        i += stride
    return out


def extract_tail_for_overlap(text: str, overlap_tokens: int, enc: tiktoken.Encoding) -> str:
    if overlap_tokens <= 0 or not text.strip():
        return ""
    sents = split_into_sentences(text)
    if len(sents) > 1:
        acc: List[str] = []
        tok = 0
        for s in reversed(sents):
            acc.append(s)
            tok += count_tokens(s, enc)
            if tok >= overlap_tokens:
                break
        return "\n\n".join(reversed(acc))
    ids = enc.encode(text)
    if len(ids) <= overlap_tokens:
        return text
    return enc.decode(ids[-overlap_tokens:])


def merge_chunks_with_overlap(
    chunks: List[str],
    overlap_tokens: int,
    max_tokens: int,
    enc: tiktoken.Encoding,
) -> List[str]:
    if not chunks or overlap_tokens <= 0:
        return chunks
    out: List[str] = [chunks[0]]
    for nxt in chunks[1:]:
        tail = extract_tail_for_overlap(out[-1], overlap_tokens, enc)
        merged = f"{tail}\n\n{nxt}".strip() if tail else nxt
        if count_tokens(merged, enc) > max_tokens:
            merged_ids = enc.encode(merged)[:max_tokens]
            merged = enc.decode(merged_ids)
        out.append(merged)
    return out


def pack_sentences_no_overlap(
    sentences: List[str],
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
    enc: tiktoken.Encoding,
) -> List[str]:
    chunks: List[str] = []
    i = 0
    n = len(sentences)
    while i < n:
        current: List[str] = []
        tok = 0
        j = i
        while j < n:
            st = sentences[j]
            piece_tok = count_tokens(st, enc)
            if not current:
                if piece_tok > max_tokens:
                    for sub in split_tokens_hard(st, max_tokens, overlap_tokens, enc):
                        chunks.append(sub)
                    j += 1
                    i = j
                    break
                current.append(st)
                tok = piece_tok
                j += 1
                if tok >= target_tokens:
                    break
                continue
            sep = count_tokens("\n\n", enc)
            if tok + sep + piece_tok > max_tokens:
                break
            current.append(st)
            tok += sep + piece_tok
            j += 1
            if tok >= target_tokens:
                break
        if current:
            chunks.append("\n\n".join(current))
        if j >= n and not current:
            break
        if not current:
            i = max(i + 1, j)
            continue
        i = j
    return chunks


def chunk_paragraphs(
    paragraphs: List[str],
    target_tokens: int = TARGET_CHUNK_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
    min_tokens: int = MIN_CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    enc: Optional[tiktoken.Encoding] = None,
) -> List[str]:
    if not paragraphs:
        return []

    enc = enc or get_token_encoder()

    def count_joined(parts: List[str]) -> int:
        if not parts:
            return 0
        return count_tokens("\n\n".join(parts), enc)

    chunks: List[str] = []
    current: List[str] = []

    for para in paragraphs:
        para_tokens = count_tokens(para, enc)

        if para_tokens > max_tokens:
            if current:
                chunks.append("\n\n".join(current).strip())
                current = []

            sents = split_into_sentences(para)
            if len(sents) <= 1:
                for piece in split_tokens_hard(para, max_tokens, overlap_tokens, enc):
                    chunks.append(piece)
            else:
                packed = pack_sentences_no_overlap(
                    sents, target_tokens, max_tokens, overlap_tokens, enc
                )
                packed = merge_chunks_with_overlap(packed, overlap_tokens, max_tokens, enc)
                chunks.extend(packed)
            continue

        projected = count_joined(current + [para]) if current else para_tokens

        if current and projected > max_tokens:
            chunks.append("\n\n".join(current).strip())
            current = [para]
        else:
            current.append(para)

    if current:
        chunks.append("\n\n".join(current).strip())

    merged: List[str] = []
    for chunk in chunks:
        if merged and count_tokens(chunk, enc) < min_tokens:
            merged[-1] = f"{merged[-1]}\n\n{chunk}".strip()
        else:
            merged.append(chunk)

    return [c for c in merged if c]


def build_chunks_from_sections(sections: List[Section]) -> List[Chunk]:
    chunks: List[Chunk] = []

    for section in sections:
        paragraphs = split_paragraphs(section.body)
        text_chunks = chunk_paragraphs(paragraphs)

        for i, text_chunk in enumerate(text_chunks, start=1):
            chunks.append(
                Chunk(
                    section_title=section.heading_text,
                    section_level=section.heading_level,
                    chunk_index_in_section=i,
                    text=text_chunk,
                )
            )

    return chunks


# =========================
# 重複抑制
# =========================

def is_near_duplicate(text_a: str, text_b: str, threshold: float = NEAR_DUPLICATE_SIMILARITY) -> bool:
    return SequenceMatcher(None, text_a, text_b).ratio() >= threshold


def dedupe_chunks_within_document(chunks: List[Chunk]) -> List[Chunk]:
    """
    同一文書内の完全重複・近似重複を軽く抑制する。
    完全一致は除外し、近似重複は直近N件とのみ比較して過剰な計算を避ける。
    """
    deduped: List[Chunk] = []
    seen_keys: Set[str] = set()
    recent_norm_texts: List[str] = []

    for chunk in chunks:
        norm = normalize_text_for_dedupe(chunk.text)
        if not norm:
            continue

        dedupe_key = md5_short(norm, 16)
        if dedupe_key in seen_keys:
            continue

        near_dup = False
        for prev in recent_norm_texts[-MAX_NEAR_DUPLICATE_COMPARISONS:]:
            if is_near_duplicate(norm, prev):
                near_dup = True
                break
        if near_dup:
            continue

        seen_keys.add(dedupe_key)
        recent_norm_texts.append(norm)
        deduped.append(chunk)

    return deduped


# =========================
# Embedding text
# =========================

def normalize_keywords_for_embedding(keywords: str) -> str:
    if not keywords:
        return ""
    tmp = keywords.replace("、", ",")
    parts = [p.strip() for p in tmp.split(",") if p.strip()]
    return ", ".join(parts)


def build_embedding_text(metadata: Dict[str, str], chunk: Chunk) -> str:
    parts = [
        f"研究タイトル: {metadata.get('title', '')}",
        f"著者名: {metadata.get('authors', '')}",
        f"提出年度: {metadata.get('year', '')}",
        f"形式: {metadata.get('format', '')}",
        f"研究分野: {metadata.get('genre', '')}",
        f"キーワード: {normalize_keywords_for_embedding(metadata.get('keywords', ''))}",
        f"資料ファイル名: {metadata.get('file_name', '')}",
        f"セクション: {chunk.section_title}",
        "",
        "本文:",
        chunk.text,
    ]
    return "\n".join(parts).strip()


# =========================
# Embedding API
# =========================

def embed_texts(texts: List[str]) -> List[List[float]]:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()
    return data["embeddings"]


# =========================
# Chroma
# =========================

def make_chunk_id(doc_id: str, global_chunk_index: int, chunk_text: str) -> str:
    digest = md5_short(chunk_text)
    return f"{doc_id}_chunk{global_chunk_index:03d}_{digest}"


def recreate_collection(client: chromadb.PersistentClient, collection_name: str):
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    return client.create_collection(name=collection_name)


# =========================
# Main
# =========================

def main() -> None:
    rows = load_rows(TSV_PATH)
    archive_index = build_archive_index(ARCHIVES_DIR)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = recreate_collection(client, COLLECTION_NAME)

    total_docs = 0
    total_chunks = 0
    total_rows_skipped = 0
    total_chunks_before_dedupe = 0
    total_chunks_removed_by_dedupe = 0

    for row_idx, row in enumerate(tqdm(rows, desc="Indexing"), start=1):
        source_filenames = split_filenames(row.get(SOURCE_FILENAME_COL, ""))
        if not source_filenames:
            total_rows_skipped += 1
            continue

        title = row.get("研究タイトル", "") or ""
        authors = row.get("著者名", "") or ""
        year = row.get("提出年度", "") or ""
        fmt = row.get("形式", "") or ""
        genre = row.get("研究分野", "") or ""
        keywords = row.get("キーワード", "") or ""
        source_url = row.get("source_url", "") or ""
        drive_file_id = row.get("drive_file_id", "") or ""

        project_id = make_project_id(title=title, authors=authors, year=year)
        author_head = get_head_author(authors)
        doc_priority = get_doc_priority(fmt)

        for file_index, source_filename in enumerate(source_filenames, start=1):
            file_path = find_archive_file(source_filename, archive_index)
            if file_path is None:
                print(f"[MISS] row={row_idx} file='{source_filename}'")
                continue

            if file_path.suffix.lower() != ".pdf":
                print(f"[SKIP_NON_PDF] row={row_idx} file='{source_filename}'")
                continue

            try:
                md = get_markdown_for_pdf(file_path, use_cache=True)
                md = clean_markdown(md)
                md = ensure_fallback_heading(md, fallback_title=title or file_path.stem)
                sections = parse_sections_from_markdown(md)
                chunks = build_chunks_from_sections(sections)
            except Exception as e:
                print(f"[ERROR] row={row_idx} file='{source_filename}' parse failed: {e}")
                continue

            if not chunks:
                print(f"[SKIP_EMPTY] row={row_idx} file='{source_filename}'")
                continue

            total_chunks_before_dedupe += len(chunks)
            chunks = dedupe_chunks_within_document(chunks)
            total_chunks_removed_by_dedupe += (total_chunks_before_dedupe - total_chunks - total_chunks_removed_by_dedupe)

            if not chunks:
                print(f"[SKIP_ALL_DEDUPED] row={row_idx} file='{source_filename}'")
                continue

            doc_id = f"row{row_idx:04d}_file{file_index:02d}"

            base_metadata = {
                "project_id": project_id,
                "doc_id": doc_id,
                "file_name": source_filename,
                "file_path": str(file_path),
                "title": title,
                "authors": authors,
                "author_head": author_head,
                "year": sanitize_metadata_value(year),
                "format": fmt,
                "doc_priority": doc_priority,
                "genre": genre,
                "keywords": keywords,
                "source_url": source_url,
                "drive_file_id": drive_file_id,
            }

            embedding_texts: List[str] = []
            documents: List[str] = []
            metadatas: List[Dict[str, str]] = []
            ids: List[str] = []

            for global_chunk_index, chunk in enumerate(chunks, start=1):
                chunk_id = make_chunk_id(doc_id, global_chunk_index, chunk.text)
                dedupe_norm = normalize_text_for_dedupe(chunk.text)
                dedupe_key = md5_short(dedupe_norm, 16)

                metadata = {
                    **base_metadata,
                    "chunk_index": global_chunk_index,
                    "section_title": chunk.section_title,
                    "section_level": chunk.section_level,
                    "chunk_index_in_section": chunk.chunk_index_in_section,
                    "dedupe_key": dedupe_key,
                }
                metadata = {k: sanitize_metadata_value(v) for k, v in metadata.items()}

                embedding_text = build_embedding_text(base_metadata, chunk)

                ids.append(chunk_id)
                documents.append(chunk.text)
                metadatas.append(metadata)
                embedding_texts.append(embedding_text)

            try:
                embeddings = embed_texts(embedding_texts)
            except Exception as e:
                print(f"[ERROR] row={row_idx} file='{source_filename}' embed failed: {e}")
                continue

            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

            total_docs += 1
            total_chunks += len(chunks)

    print("=== DONE ===")
    print(f"rows_total={len(rows)}")
    print(f"rows_skipped={total_rows_skipped}")
    print(f"docs_indexed={total_docs}")
    print(f"chunks_indexed={total_chunks}")
    print(f"chunks_before_dedupe={total_chunks_before_dedupe}")
    print(f"chunks_removed_by_dedupe={total_chunks_removed_by_dedupe}")


if __name__ == "__main__":
    main()