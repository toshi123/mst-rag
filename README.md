# mst-rag

RAG（Retrieval-Augmented Generation）向けのスクリプトと Streamlit UI を含むリポジトリです。

## 別環境で `git clone` したあとにやること

### 1. Conda 環境の作成と有効化

このプロジェクトは `environment.yml` で Conda 環境を定義しています（環境名は `mst-rag-admin`）。

```bash
cd mst-rag
conda env create -f environment.yml
conda activate mst-rag-admin
```

環境を更新したい場合:

```bash
conda env update -f environment.yml --prune
```

### 2. リポジトリに含まれないものの準備（`.gitignore` との対応）

次のパスやファイルは **Git にコミットされません**。クローン直後は存在しないため、必要に応じてローカルで用意してください。

| 対象 | 内容 |
|------|------|
| `data/` | データ置き場。インデックス構築や運用で使う場合は、手元でディレクトリを作成するか、別途データを配置します。 |
| `.env` / `.env.*` | シェルやツール用のローカル環境変数ファイル。必要ならプロジェクトルートに作成し、変数を設定します（下記「環境変数」を参照）。 |
| `service_account.json` / `credentials.json` | Google API 用の認証ファイル。`scripts/fill_drive_links_to_tsv.py` などで使う場合のみ配置します。 |
| `.venv/` / `venv/` / `env/` | 仮想環境は各自のマシンで作成する前提です。本リポジトリは Conda を推奨していますが、venv を使う場合も同様に Git 対象外です。 |

機密情報やローカル専用パスはリポジトリに入れない設計のため、**チーム内で別ルート（共有ストレージやドキュメント）で受け渡しする**想定です。

### 3. 環境変数（アプリ・スクリプトが参照）

用途に応じて設定してください（未設定時はコード内のデフォルトが使われます）。

**RAG / Streamlit（`app.py` / `app/streamlit_app.py` など）**

- `CHROMA_DIR` — Chroma の永続ディレクトリ。デフォルトは開発者マシン向けの絶対パスになっているため、**別環境では必ず自分のパスに合わせて設定**してください。
- `CHROMA_COLLECTION` — コレクション名（既定: `esa_posts`）。
- `OLLAMA_HOST` — Ollama のベース URL（既定: `http://localhost:11434`）。
- `OLLAMA_EMBED_MODEL` / `OLLAMA_CHAT_MODEL` — 使用するモデル名。
- **`RERANK_ENABLED`** — Streamlit の検索設定で「クロスエンコーダでリランク」を**既定でオンにしたい場合**に設定します。`1` のときのみ有効です。未設定または `1` 以外のときは、起動時のチェックボックスはオフです。リランクには `sentence-transformers`（およびモデル初回ダウンロード）が必要です。
- `RERANK_MODEL` — リランク用 CrossEncoder のモデル名（省略時はコード側の既定、例: `BAAI/bge-reranker-base`）。CLI の `query_index.py` / `ask_rag.py` の `--rerank` 利用時にも参照されます。
- **`HF_TOKEN`** — Hugging Face の[アクセストークン](https://huggingface.co/settings/tokens)。リランクで Hub からモデルを取得する際、**未設定だと「unauthenticated requests」などの警告**が出ることがあり、認証済みリクエストにするとレート制限緩和・ダウンロードの安定に役立ちます。必須ではありません。

**esa 連携スクリプト（`scripts/sync_esa_to_chroma.py`, `scripts/debug_esa_mcp.py`）**

- `ESA_ACCESS_TOKEN` — 必須（未設定だと該当スクリプトは起動時にエラーになります）。
- `ESA_TEAM` — チーム名が必要な場合に設定。

シェルで一時的に設定する例:

```bash
export CHROMA_DIR="/path/to/your/chroma_db"
export OLLAMA_HOST="http://localhost:11434"
# Streamlit でリランクを既定オンにする（起動前に設定）
export RERANK_ENABLED=1
```

`.env` を使う場合は、お使いのシェルや direnv などで読み込む運用にしてください（ファイル自体は Git されません）。

**リランク初回時のターミナル表示について**

- `BAAI/bge-reranker-base` などを初めて読み込むと、重みの読み込み進捗や `LOAD REPORT`（`roberta.embeddings.position_ids` が `UNEXPECTED` など）が表示されることがあります。後者はクロスエンコーダ用途でよくある表示で、**多くの場合は無視して問題ありません**。
- 未認証の警告を減らしたい場合は上記 **`HF_TOKEN`** を設定してください。`rag_retrieval.py` では Hub の冗長ログを抑える設定も入れていますが、ライブラリのバージョンによっては一部メッセージが残ることがあります。

### 4. 外部サービス・データの前提

- **Ollama**: 埋め込み・チャットで利用する場合、ローカルまたは指定ホストで Ollama が起動している必要があります。
- **Chroma DB**: `CHROMA_DIR` に既存のインデックスが無い場合は、先に `scripts/build_index.py` や `scripts/sync_esa_to_chroma.py` などで構築する必要があります。
  - DBの構築については [build_db.md](./docs/build_db.md) を参照のこと

### 5. アプリの起動例

プロジェクト構成に応じて、例えば次のように起動できます（実際のエントリポイントは用途に合わせて選択してください）。

```bash
conda activate mst-rag-admin
streamlit run app/streamlit_app.py
# または
streamlit run app.py
```

---

## `environment.yml` とコードの対応（依存関係のメモ）

- リポジトリ内の Python から import されているサードパーティは `environment.yml` に揃えています。
- **`mcp`**（`scripts/sync_esa_to_chroma.py`, `scripts/debug_esa_mcp.py`）は pip 経由で記載されています。
- **`pandas`** と **`pyyaml`** は現状の `.py` からは参照されていませんが、Conda 側の便利依存として残している場合があります。最小構成にしたい場合は `environment.yml` から削除して問題ない可能性があります。

依存を追加したら、`environment.yml` を更新し、この README の該当箇所も合わせて更新してください。
