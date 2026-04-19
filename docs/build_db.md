# RAG データベース構築手順

このリポジトリは、Chroma DB などの生成済み RAG データを含まない状態を前提としています。  
clone 後、各ユーザーが手元で PDF 資料および esa 記事から RAG 用データベースを構築してください。

---

## 前提

- Python 環境が用意されていること
- Ollama がインストール済みであること
- 埋め込みモデルが pull 済みであること
- esa を取り込む場合は、esa MCP または esa 取得に必要な認証情報・設定が済んでいること
- PDF 元データおよび `data/research_list.tsv` が所定の場所に配置されていること

---

## ディレクトリ想定

```text
project-root/
├── app/
├── scripts/
├── data/
│   ├── archives/
│   ├── chroma/
│   ├── processed_markdown/
│   └── research_list.tsv
├── state/
└── ...
````

* `data/archives/` : PDF 置き場
* `data/research_list.tsv` : PDF 台帳
* `data/chroma/` : Chroma DB 保存先
* `data/processed_markdown/` : PDF から抽出した Markdown キャッシュ
* `state/` : esa 同期状態などの管理用

---

## 0. リポジトリ取得

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

---

## 1. Python 環境を作成・有効化

### conda を使う場合

```bash
conda env create -f environment.yml
conda activate <ENV_NAME>
```

既存環境を更新する場合:

```bash
conda env update -f environment.yml --prune
conda activate <ENV_NAME>
```

### venv を使う場合

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

---

## 2. Ollama を起動し、必要モデルを pull する

### Ollama 起動確認

```bash
ollama list
```

### 埋め込みモデルの取得

```bash
ollama pull embeddinggemma
```

### 生成モデルの取得（例）

```bash
ollama pull gemma4:e4b
```

### 動作確認

```bash
ollama run embeddinggemma
```

> `embeddinggemma` は通常 generate 用ではないため、実運用では埋め込み API 経由で使われます。
> 少なくとも `ollama list` にモデル名が見えていれば OK です。

---

## 3. 必要ディレクトリを作成

```bash
mkdir -p data/archives
mkdir -p data/chroma
mkdir -p data/processed_markdown
mkdir -p state
```

---

## 4. PDF 資料を配置

PDF を `data/archives/` に置いてください。
また、対応する台帳ファイル `data/research_list.tsv` を用意してください。

例:

```bash
ls data/archives
ls data/research_list.tsv
```

---

## 5. PDF 由来の Chroma DB を構築

PDF 側の collection 名は `mst_research` を想定しています。
`build_index.py` が以下を行います。

* PDF を Markdown に変換
* 見出し付き section を抽出
* 段落ベースで chunk 化
* metadata を付与
* Chroma DB に投入

実行例:

```bash
python scripts/build_index.py \
  --tsv data/research_list.tsv \
  --pdf-dir data/archives \
  --markdown-dir data/processed_markdown \
  --chroma-dir data/chroma \
  --collection mst_research
```

---

## 6. esa 由来の Chroma DB を構築

esa 側の collection 名は `esa_posts` を想定しています。

`sync_esa_to_chroma.py` が以下を行います。

* esa から記事取得
* 本文・見出し・category・tags などを整理
* chunk 化
* Chroma DB に投入
* state を更新

実行例:

```bash
python scripts/sync_esa_to_chroma.py \
  --chroma-dir data/chroma \
  --collection esa_posts \
  --state-file state/esa_sync_state.json \
  --bootstrap
```

---

## 7. Chroma DB の構築結果を確認

### collection ができているか確認

プロジェクトに確認用スクリプトがある場合:

```bash
python scripts/query_index.py --query "test" --source pdf
python scripts/query_index.py --query "test" --source esa
```

### あるいは Streamlit UI で確認

```bash
streamlit run app/streamlit_app.py
```

---

## 8. うまくいかないときの確認ポイント

### Ollama モデルが入っていない

```bash
ollama list
```

* `embeddinggemma`
* 使用したい生成モデル（例: `gemma4:e4b`）

が見えているか確認してください。

### `esa_posts` が空

* `state/esa_sync_state.json` が残っていて、空なのに同期済み扱いになっていないか
* `--bootstrap` を付けて再同期したか
* esa 接続設定や認証が通っているか

### PDF が拾われない

* `data/research_list.tsv` のパスが正しいか
* `data/archives/` に PDF があるか
* TSV 内のファイル名と実ファイル名が対応しているか

---

## 9. よく使う再構築コマンド

### PDF 側だけ再構築

```bash
rm -rf data/chroma
mkdir -p data/chroma

python scripts/build_index.py \
  --tsv data/research_list.tsv \
  --pdf-dir data/archives \
  --markdown-dir data/processed_markdown \
  --chroma-dir data/chroma \
  --collection mst_research
```

### esa 側だけ再同期

```bash
python scripts/sync_esa_to_chroma.py \
  --chroma-dir data/chroma \
  --collection esa_posts \
  --state-file state/esa_sync_state.json \
  --bootstrap
```

### 全再構築

```bash
rm -rf data/chroma
rm -rf data/processed_markdown
rm -f state/esa_sync_state.json

mkdir -p data/chroma
mkdir -p data/processed_markdown
mkdir -p state

python scripts/build_index.py \
  --tsv data/research_list.tsv \
  --pdf-dir data/archives \
  --markdown-dir data/processed_markdown \
  --chroma-dir data/chroma \
  --collection mst_research

python scripts/sync_esa_to_chroma.py \
  --chroma-dir data/chroma \
  --collection esa_posts \
  --state-file state/esa_sync_state.json \
  --bootstrap
```

---

## 10. 最後に UI を起動

```bash
streamlit run app/streamlit_app.py
```

起動後、以下を確認してください。

* `source=pdf` で PDF 検索が動く
* `source=esa` で esa 検索が動く
* `source=all` で両方検索できる
* 回答生成時に Ollama エラーが出ない

---

## 備考

* DB を Git 管理しない前提では、`data/chroma/`, `data/processed_markdown/`, `state/esa_sync_state.json` などは `.gitignore` に入れておくのがおすすめです。
* 大きい生成モデルが厳しい場合は、Streamlit UI の生成モデル欄で軽量モデルに切り替えてください。
* retrieval の品質評価は、よく使うクエリをいくつか固定して確認するのがおすすめです。

