#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import html
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests
import streamlit as st

# scripts/ 配下のモジュールを import できるようにする
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from rag_retrieval import retrieve_hits, unique_source_entries, build_context  # noqa: E402


OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_GENERATE_MODEL = "gemma4:e4b"


TEXTS = {
    "ja": {
        "page_title": "MST RAG Viewer",
        "title": "MST Research RAG",
        "caption": "研究資料PDFとesa記事を横断して検索・要約するローカルUI",
        "hero_title": "何を調べますか？",
        "hero_subtitle": "研究資料アーカイブと esa 記事アーカイブを横断して検索できます。",
        "settings": "設定",
        "question": "質問",
        "default_query": "脳に関する研究について教えて",
        "query_placeholder": "例: 脳に関する研究について教えて",
        "search_box_label": "検索",
        "source": "検索対象",
        "metadata_filter": "Metadata filter",
        "year": "提出年度",
        "format": "形式",
        "genre": "研究分野",
        "project_id": "project_id",
        "retrieval": "Retrieval",
        "distance_cutoff_checkbox": "distance cutoff を使う",
        "reranker_checkbox": "クロスエンコーダでリランク（要 sentence-transformers）",
        "generation": "Generation",
        "model": "生成モデル",
        "show_context": "コンテキストを表示",
        "search": "検索",
        "generate": "RAG回答を生成",
        "search_and_generate": "検索して回答する",
        "searching": "検索しています...",
        "generating": "回答を生成しています...",
        "enter_question": "質問を入力してください。",
        "search_error": "検索中にエラーが発生しました",
        "generate_error": "生成中にエラーが発生しました",
        "answer_tab": "回答",
        "results_tab": "検索結果",
        "refs_tab": "参照資料",
        "rag_answer": "RAG回答",
        "not_generated": "まだ回答は生成されていません。上の検索バーの「RAG回答を生成」または「検索して回答する」を押してください。",
        "not_searched": "まだ検索していません。",
        "no_refs": "参照資料はまだありません。",
        "refs_none": "参照資料はありません。",
        "references": "参照資料",
        "open_pdf": "資料を開く",
        "open_esa": "esa記事を開く",
        "open_ref": "参照先を開く",
        "chunk_expander": "チャンク本文を見る",
        "used_context": "生成に使ったコンテキスト",
        "language_subheader": "Language",
        "language_radio": "UI language / 回答言語",
        "source_subheader": "Source",
        "raw_hits": "Raw hits",
        "final_hits": "Final hits",
        "source_metric": "Source",
        "pdf_badge": "📄 PDF",
        "esa_badge": "📝 esa",
        "unknown_badge": "❓ unknown",
        "topbar_hint": "検索条件は左のサイドバーで調整できます。",
        "new_search": "新しい検索",
        "search_only": "検索のみ",
        "answer_only": "RAG回答を生成",
        "search_and_answer": "検索して回答する",
        "search_results_header": "検索結果",
    },
    "en": {
        "page_title": "MST RAG Viewer",
        "title": "MST Research RAG",
        "caption": "Local UI for searching and summarizing research PDFs and esa articles",
        "hero_title": "What would you like to explore?",
        "hero_subtitle": "Search across the research archive and the esa article archive.",
        "settings": "Settings",
        "question": "Question",
        "default_query": "Tell me about research related to the brain.",
        "query_placeholder": "e.g. Tell me about research related to the brain.",
        "search_box_label": "Search",
        "source": "Search target",
        "metadata_filter": "Metadata filter",
        "year": "Year",
        "format": "Format",
        "genre": "Research field",
        "project_id": "project_id",
        "retrieval": "Retrieval",
        "distance_cutoff_checkbox": "Use distance cutoff",
        "reranker_checkbox": "Rerank with cross-encoder (requires sentence-transformers)",
        "generation": "Generation",
        "model": "Generation model",
        "show_context": "Show context",
        "search": "Search",
        "generate": "Generate RAG Answer",
        "search_and_generate": "Search and Answer",
        "searching": "Searching...",
        "generating": "Generating answer...",
        "enter_question": "Please enter a question.",
        "search_error": "An error occurred during retrieval",
        "generate_error": "An error occurred during generation",
        "answer_tab": "Answer",
        "results_tab": "Search Results",
        "refs_tab": "References",
        "rag_answer": "RAG Answer",
        "not_generated": "No answer has been generated yet. Use the top search bar to generate one.",
        "not_searched": "No search has been run yet.",
        "no_refs": "No references yet.",
        "refs_none": "No references.",
        "references": "References",
        "open_pdf": "Open document",
        "open_esa": "Open esa article",
        "open_ref": "Open reference",
        "chunk_expander": "Show chunk text",
        "used_context": "Context used for generation",
        "language_subheader": "Language",
        "language_radio": "UI language / Answer language",
        "source_subheader": "Source",
        "raw_hits": "Raw hits",
        "final_hits": "Final hits",
        "source_metric": "Source",
        "pdf_badge": "📄 PDF",
        "esa_badge": "📝 esa",
        "unknown_badge": "❓ unknown",
        "topbar_hint": "You can adjust search settings in the left sidebar.",
        "new_search": "New search",
        "search_only": "Search only",
        "answer_only": "Generate answer",
        "search_and_answer": "Search and Answer",
        "search_results_header": "Search Results",
    },
}


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


def build_prompt(user_query: str, context: str, ui_language: str) -> str:
    lang_map = {
        "ja": {
            "system": "あなたは、学校内の研究資料アーカイブとesa記事アーカイブをもとに回答するアシスタントです。",
            "rules": [
                "以下のコンテキストだけを根拠として回答してください。",
                "根拠が足りない場合は、足りないと明記してください。",
                "推測で断定しないでください。",
                "回答は必ず日本語で書いてください。",
                "中国語・英語・その他の言語を混ぜず、日本語のみで回答してください。",
                "質問文やコンテキストに他言語が含まれていても、回答は必ず日本語にしてください。",
                "回答の最終出力は必ず日本語のみで行ってください。",
            ],
            "policy_title": "回答方針:",
            "policy_items": [
                "要点を最初に短く述べる",
                "文脈上わかる範囲で、どの資料群に基づくかが伝わるように書く",
                "最後に「参照資料:」という見出しは付けない（別で整形表示するため）",
                "その後、背景・共通点・相違点・具体例・今後の論点まで詳しく述べる",
                "可能な限り複数の資料の内容を統合して説明する",
                "回答は短く切り上げず、十分な具体性を持たせる",
                "コンテキストにないことは言い切らない",
            ],
            "question": "質問:",
            "context": "コンテキスト:",
        },
        "en": {
            "system": "You are an assistant that answers questions based on a school research archive and an esa article archive.",
            "rules": [
                "Answer using only the context below as evidence.",
                "If the context is insufficient, explicitly say so.",
                "Do not make definitive claims based on guesswork.",
                "Your answer must be written in English.",
                "Do not mix Chinese, Japanese, or any other language into the answer.",
                "Even if the question or context includes other languages, the answer must be in English.",
                "The final answer must be written in English only.",
            ],
            "policy_title": "Answer policy:",
            "policy_items": [
                "Begin with a brief statement of the key point.",
                "Write in a way that makes it clear, as far as the context allows, which group of materials the answer is based on.",
                'Do not add a heading such as "References:" at the end (references are rendered separately).',
                "Then provide a detailed explanation covering the background, common points, differences, concrete examples, and future issues.",
                "When possible, integrate and synthesize information from multiple sources.",
                "Do not end the answer too briefly; provide sufficient specificity and detail.",
                "Do not state anything as certain if it is not supported by the context.",
            ],
            "question": "Question:",
            "context": "Context:",
        },
    }

    t = lang_map.get(ui_language, lang_map["ja"])
    rules_text = "\n".join(f"- {item}" for item in t["rules"])
    policy_text = "\n".join(f"- {item}" for item in t["policy_items"])

    return f"""{t["system"]}
{rules_text}

{t["policy_title"]}
{policy_text}

{t["question"]}
{user_query}

{t["context"]}
{context}
""".strip()


def source_badge(source_type: str, ui_language: str) -> str:
    t = TEXTS[ui_language]
    if source_type == "pdf":
        return t["pdf_badge"]
    if source_type == "esa":
        return t["esa_badge"]
    return t["unknown_badge"]


def render_reference_list(refs: List[Dict[str, str]], ui_language: str) -> None:
    t = TEXTS[ui_language]

    if not refs:
        st.info(t["refs_none"])
        return

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
            prefix = "📄"
        elif source_type == "esa":
            label_parts = []
            if ref.get("title"):
                label_parts.append(ref["title"])
            if ref.get("category"):
                label_parts.append(f"category={ref['category']}")
            if ref.get("heading"):
                label_parts.append(f"heading={ref['heading']}")
            label = " / ".join(label_parts) if label_parts else "(esa)"
            prefix = "📝"
        else:
            label = ref.get("title") or "(untitled)"
            prefix = "❓"

        source_url = ref.get("source_url") or ""
        if source_url:
            st.markdown(f"- {prefix} [{label}]({source_url})")
        else:
            st.markdown(f"- {prefix} {label}")


def render_hit_card(hit: Dict[str, Any], idx: int, ui_language: str) -> None:
    t = TEXTS[ui_language]
    meta = hit["metadata"]
    source_type = hit.get("source_type") or meta.get("source_type") or "unknown"
    collection_name = hit.get("collection_name") or meta.get("collection_name") or ""

    with st.container(border=True):
        title = meta.get("title", "") or "(untitled)"
        st.markdown(f"**[{idx}] {source_badge(source_type, ui_language)} {title}**")

        cols = st.columns(5)
        cols[0].write(f"chroma_dist: {hit.get('chroma_distance', hit.get('distance', ''))}")
        cols[1].write(
            f"rerank: {hit['rerank_score']:.4f}"
            if hit.get("rerank_score") is not None
            else "rerank: —"
        )
        cols[2].write(f"collection: {collection_name}")
        cols[3].write(f"source: {source_type}")
        cols[4].write(f"rank_raw: {hit.get('rank_raw', '')}")

        if source_type == "pdf":
            st.write(f"format/year: {meta.get('format', '')} / {meta.get('year', '')}")
            st.write(f"genre: {meta.get('genre', '')}")
            st.write(f"section: {meta.get('section_title', '')}")
            st.write(f"authors: {meta.get('authors', '')}")
            st.write(f"project_id: {meta.get('project_id', '')}")
            st.write(f"doc_id: {meta.get('doc_id', '')}")
            if meta.get("source_url"):
                st.markdown(f"[{t['open_pdf']}]({meta['source_url']})")

        elif source_type == "esa":
            st.write(f"category: {meta.get('category', '')}")
            st.write(f"heading: {meta.get('heading', '') or meta.get('section_title', '')}")
            st.write(f"tags: {meta.get('tags', '')}")
            if meta.get("url") or meta.get("source_url"):
                st.markdown(f"[{t['open_esa']}]({meta.get('url') or meta.get('source_url')})")

        else:
            if meta.get("source_url"):
                st.markdown(f"[{t['open_ref']}]({meta['source_url']})")

        with st.expander(t["chunk_expander"]):
            st.write(hit["document"])


def run_retrieval(
    query: str,
    source: str,
    top_k: int,
    initial_k: int,
    use_distance_cutoff: bool,
    distance_cutoff: float,
    max_per_doc: int,
    max_per_project: int,
    max_per_url: int,
    year: str,
    fmt: str,
    genre: str,
    project_id: str,
    use_cross_encoder_rerank: bool,
) -> Dict[str, Any]:
    return retrieve_hits(
        query=query,
        source=source,
        top_k=top_k,
        initial_k=initial_k,
        distance_cutoff=distance_cutoff if use_distance_cutoff else None,
        max_per_doc=max_per_doc,
        max_per_project=max_per_project,
        max_per_url=max_per_url,
        year=year or None,
        fmt=fmt or None,
        genre=genre or None,
        project_id=project_id or None,
        use_cross_encoder_rerank=use_cross_encoder_rerank,
    )


def execute_search(
    query: str,
    source: str,
    top_k: int,
    initial_k: int,
    use_distance_cutoff: bool,
    distance_cutoff: float,
    max_per_doc: int,
    max_per_project: int,
    max_per_url: int,
    year: str,
    fmt: str,
    genre: str,
    project_id: str,
    use_cross_encoder_rerank: bool,
):
    result = run_retrieval(
        query=query,
        source=source,
        top_k=top_k,
        initial_k=initial_k,
        use_distance_cutoff=use_distance_cutoff,
        distance_cutoff=distance_cutoff,
        max_per_doc=max_per_doc,
        max_per_project=max_per_project,
        max_per_url=max_per_url,
        year=year,
        fmt=fmt,
        genre=genre,
        project_id=project_id,
        use_cross_encoder_rerank=use_cross_encoder_rerank,
    )
    final_hits = result["final_hits"]
    context = build_context(final_hits)
    refs = unique_source_entries(final_hits)

    st.session_state.last_result = result
    st.session_state.last_context = context
    st.session_state.last_refs = refs
    st.session_state.has_searched = True


def execute_answer(query: str, model: str, ui_language: str):
    context = st.session_state.last_context or ""
    prompt = build_prompt(query, context, ui_language)
    answer = generate_answer(prompt, model=model)
    st.session_state.last_answer = answer


def render_search_center(query_key: str, ui_language: str) -> tuple[str, bool, bool, bool]:
    t = TEXTS[ui_language]

    st.markdown(
        """
        <style>
        .hero-search-wrap {
            max-width: 820px;
            margin: 10vh auto 0 auto;
            text-align: center;
        }
        .hero-search-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }
        .hero-search-subtitle {
            color: #666;
            margin-bottom: 1.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="hero-search-wrap">', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-search-title">{html.escape(t["hero_title"])}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-search-subtitle">{html.escape(t["hero_subtitle"])}</div>', unsafe_allow_html=True)

    with st.form("center_search_form", clear_on_submit=False):
        query = st.text_input(
            t["question"],
            value=st.session_state.get(query_key, t["default_query"]),
            placeholder=t["query_placeholder"],
            key=f"{query_key}_center_input",
            label_visibility="collapsed",
        )
        c1, c2, c3 = st.columns([1, 1, 1])
        run_search = c1.form_submit_button(t["search_only"], use_container_width=True)
        run_answer = c2.form_submit_button(t["answer_only"], use_container_width=True)
        run_both = c3.form_submit_button(t["search_and_answer"], use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
    return query, run_search, run_answer, run_both


def render_search_topbar(query_key: str, ui_language: str) -> tuple[str, bool, bool, bool]:
    t = TEXTS[ui_language]

    st.markdown(
        """
        <style>
        .top-search-note {
            color: #666;
            margin-top: -0.3rem;
            margin-bottom: 0.6rem;
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"<div class='top-search-note'>{html.escape(t['topbar_hint'])}</div>", unsafe_allow_html=True)

    with st.form("top_search_form", clear_on_submit=False):
        c1, c2, c3, c4 = st.columns([8, 1.4, 1.8, 2.0])

        query = c1.text_input(
            t["search_box_label"],
            value=st.session_state.get(query_key, t["default_query"]),
            placeholder=t["query_placeholder"],
            key=f"{query_key}_top_input",
            label_visibility="collapsed",
        )
        run_search = c2.form_submit_button(t["search_only"], use_container_width=True)
        run_answer = c3.form_submit_button(t["answer_only"], use_container_width=True)
        run_both = c4.form_submit_button(t["search_and_answer"], use_container_width=True)

    return query, run_search, run_answer, run_both


def main() -> None:
    if "ui_language" not in st.session_state:
        st.session_state.ui_language = "ja"
    if "has_searched" not in st.session_state:
        st.session_state.has_searched = False
    if "current_query" not in st.session_state:
        st.session_state.current_query = TEXTS["ja"]["default_query"]
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = None
    if "last_context" not in st.session_state:
        st.session_state.last_context = None
    if "last_refs" not in st.session_state:
        st.session_state.last_refs = []

    st.set_page_config(
        page_title=TEXTS[st.session_state.ui_language]["page_title"],
        page_icon="🔎",
        layout="wide",
    )

    with st.sidebar:
        st.subheader("Language")
        ui_language = st.radio(
            "UI language / 回答言語",
            options=["ja", "en"],
            index=0 if st.session_state.ui_language == "ja" else 1,
            format_func=lambda x: "日本語" if x == "ja" else "English",
            horizontal=True,
        )
        st.session_state.ui_language = ui_language

    t = TEXTS[ui_language]

    st.title(t["title"])
    st.caption(t["caption"])

    with st.sidebar:
        st.header(t["settings"])

        st.subheader(t["source_subheader"])
        source = st.radio(
            t["source"],
            options=["all", "pdf", "esa"],
            index=0,
            horizontal=True,
        )

        st.subheader(t["metadata_filter"])
        year = st.text_input(t["year"], value="")
        fmt = st.selectbox(t["format"], options=["", "Report", "Poster", "Title"], index=0)
        genre = st.text_input(t["genre"], value="")
        project_id = st.text_input(t["project_id"], value="")

        st.subheader(t["retrieval"])
        top_k = st.slider("top_k", min_value=1, max_value=25, value=8)
        initial_k = st.slider("initial_k", min_value=top_k, max_value=100, value=max(24, top_k))
        use_distance_cutoff = st.checkbox(t["distance_cutoff_checkbox"], value=False)
        distance_cutoff = st.number_input("distance_cutoff", min_value=0.0, max_value=10.0, value=1.2, step=0.1)
        use_cross_encoder_rerank = st.checkbox(
            t["reranker_checkbox"],
            value=os.environ.get("RERANK_ENABLED", "0") == "1",
        )
        max_per_doc = st.slider("max_per_doc", min_value=1, max_value=5, value=2)
        max_per_project = st.slider("max_per_project", min_value=1, max_value=5, value=3)
        max_per_url = st.slider("max_per_url", min_value=1, max_value=5, value=2)

        st.subheader(t["generation"])
        model = st.text_input(t["model"], value=DEFAULT_GENERATE_MODEL)
        show_context = st.checkbox(t["show_context"], value=False)

    if st.session_state.has_searched:
        query, run_search, run_answer, run_both = render_search_topbar("current_query", ui_language)
    else:
        query, run_search, run_answer, run_both = render_search_center("current_query", ui_language)

    st.session_state.current_query = query

    if run_search or run_answer or run_both:
        if not query.strip():
            st.warning(t["enter_question"])
            st.stop()

        try:
            if run_search or run_both or (run_answer and not st.session_state.last_result):
                with st.spinner(t["searching"]):
                    execute_search(
                        query=query,
                        source=source,
                        top_k=top_k,
                        initial_k=initial_k,
                        use_distance_cutoff=use_distance_cutoff,
                        distance_cutoff=distance_cutoff,
                        max_per_doc=max_per_doc,
                        max_per_project=max_per_project,
                        max_per_url=max_per_url,
                        year=year,
                        fmt=fmt,
                        genre=genre,
                        project_id=project_id,
                        use_cross_encoder_rerank=use_cross_encoder_rerank,
                    )

            if run_answer or run_both:
                with st.spinner(t["generating"]):
                    execute_answer(query=query, model=model, ui_language=ui_language)

        except Exception as e:
            action_label = t["generate_error"] if (run_answer or run_both) else t["search_error"]
            st.error(f"{action_label}: {e}")
            st.stop()

    result = st.session_state.last_result
    answer = st.session_state.last_answer
    context = st.session_state.last_context
    refs = st.session_state.last_refs

    tab1, tab2, tab3 = st.tabs([t["answer_tab"], t["results_tab"], t["refs_tab"]])

    with tab1:
        if result is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric(t["raw_hits"], len(result["raw_hits"]))
            c2.metric(t["final_hits"], len(result["final_hits"]))
            c3.metric(t["source_metric"], result.get("source", "all"))

            st.write(f"searched_collections: {result.get('searched_collections', [])}")
            if result.get("missing_collections"):
                st.warning(f"missing_collections: {result['missing_collections']}")
            if result.get("rerank_requested"):
                applied = result.get("rerank_applied")
                rm = result.get("rerank_model") or ""
                st.caption(
                    f"rerank: {'OK' if applied else 'skipped (install sentence-transformers or check model)'} "
                    f"model={rm}"
                )

        if answer:
            st.subheader(t["rag_answer"])
            st.write(answer)
        else:
            st.info(t["not_generated"])

        if show_context and context:
            with st.expander(t["used_context"]):
                st.text(context)

    with tab2:
        if result is None:
            st.info(t["not_searched"])
        else:
            st.subheader(t["search_results_header"])
            for i, hit in enumerate(result["final_hits"], start=1):
                render_hit_card(hit, i, ui_language)

    with tab3:
        if refs:
            st.subheader(t["references"])
            render_reference_list(refs, ui_language)
        else:
            st.info(t["no_refs"])


if __name__ == "__main__":
    main()