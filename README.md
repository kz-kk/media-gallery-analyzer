# Media Gallery Analyzer

[English](README_en.md) | [日本語](README.md)

AI画像解析、全文検索、セマンティック検索機能を備えたメディアギャラリービューアです。

## 機能

- 📁 **スマートメディアスキャン**: カスタマイズ可能な深度でのディレクトリ再帰スキャン
- 🤖 **AI画像解析**: LM Studioを使用した自動キャプション・タグ生成
- 🎬 **動画処理**: 動画ファイルのフレーム抽出・解析
- 🔍 **高度な検索機能**:
  - Meilisearchによる全文検索
  - Qdrantベクトル埋め込みによるセマンティック検索
  - AIによる結果再ランキング

## 前提条件

- Node.js (>=18)
- Python 3.11+
- Docker & Docker Compose (Docker Desktop https://docs.docker.com/desktop/setup/install/mac-install/)
- FFmpeg（動画処理用）
- LM Studio (>=0.3.24) 古いバージョンでは画像解析リクエストが通らない

## モデルの取得方法

- LM Studio（視覚モデル）
  - LM Studio: https://lmstudio.ai/
  - qwen2.5-vl-7b は LM Studio の Model Browser からダウンロードして読み込みます
  - SVGロゴ解析は CairoSVG により白/黒背景の2枚をラスタライズしてからVLMに送信します

- Python 依存（requirements.txt で一括）
  - CairoSVG（SVGラスタライズ用）、Pillow、requests、sentence-transformers など

- 仮想環境内 Python（自動ダウンロード）
  - `python setup_models.py` を実行すると、以下を自動ダウンロードしてキャッシュします
    - Plamo Embedding 1B（セマンティック検索用）
    - Whisper small（音声認識用）
  - 参考リンク（任意）
    - Plamo: https://huggingface.co/pfnet/plamo-embedding-1b
    - 軽量代替 all-MiniLM-L6-v2: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    - Whisper モデル一覧: https://github.com/openai/whisper#available-models

## クイックスタート

### 1. クローンとセットアップ

```bash
git clone https://github.com/kz-kk/media-gallery-analyzer.git
cd media-gallery-analyzer
cp .env.example .env
```

### 2. 依存関係のインストール

#### Node.js依存関係インストール
```bash
# （Node 18 以上を推奨）
npm install
```

#### 仮想環境にてPython依存関係インストール
```bash
# Mac 
python -m venv venv
source venv/bin/activate  

# Windows Gitbash 
python3 -m venv venv
. venv/Scripts/activate

# 依存関係install
(venv) pip install -r requirements.txt

# AIモデル(whisper・plamo)のダウンロード
(venv) python setup_models.py
```

### 3. 環境設定

`.env`ファイルを編集：

```bash
# 必須: メディアディレクトリのパスを設定
SCAN_PATH=/path/to/your/media/directory
# 仮想環境
PYTHON_VENV_PATH=./venv/bin/python

# 例 Mac
# SCAN_PATH=/Users/example/media

# 例 windows Gitbash
# SCAN_PATH=C:\Users\example\media

# LMStudioのAPI受付URL 
# API設定タブ > Settings > Serve on Local NetworkをOff
LMSTUDIO_URL=http://127.0.0.1:1234

```


### 4. サービスの開始

```bash
# Docker Desktopを起動後実行 MeilisearchとQdrantを開始
docker compose up -d

# LMStudioであらかじめマルチモーダルのqwen2.5-vl-7bをダウンロードして使えるようにして起動しておく

# アプリケーションを起動
# http://127.0.0.1:3333/
npm start
```


### 5. アプリケーションへのアクセス（ローカル）
基本はギャラリーUIにのみアクセス

| 名称 | URL | ポート/ENV | 説明 |
|------|-----|-----------|------|
| ギャラリーUI | http://127.0.0.1:3333 | `PORT=3333` | 本アプリのフロントエンド |
| Meilisearch 管理画面UI | http://127.0.0.1:24900 | 固定 24900 | 公式UI（`docker compose`で起動） |
| Meilisearch API | http://127.0.0.1:17700 | 固定 17700 → コンテナ内7700 | 検索エンジンAPI（`MEILI_URL`） |
| Qdrant ダッシュボード | http://127.0.0.1:26333/dashboard | 固定 26333 → コンテナ内6333 | ベクトルDBダッシュボード |

### Meilisearch UI 初回接続設定

Meilisearch管理画面（http://127.0.0.1:24900）を初回利用時は、以下の設定で接続してください：

- **Name**: `default`
- **Host**: `http://127.0.0.1:17700`
- **API Key**: `masterKey` (デフォルト値 - `.env`の`MEILI_MASTER_KEY`)

## 設定

### 環境変数

| 変数名 | デフォルト値 | 説明 |
|--------|-------------|------|
| `SCAN_PATH` | - | **必須** メディアディレクトリのパス |
| `PORT` | 3333 | サーバーポート |
| `LMSTUDIO_URL` | http://127.0.0.1:1234 | LM Studio API URL |
| `LMSTUDIO_MODEL` | qwen2.5-vl-7b | ビジョンモデル名 |
| `LMSTUDIO_IMAGE_MODE` | data | 画像の渡し方（data/http）|
| `MEILI_URL` | http://127.0.0.1:17700 | Meilisearch API URL |
| `QDRANT_URL` | http://127.0.0.1:26333 | Qdrant API URL |
| `EMB_MODEL` | pfnet/plamo-embedding-1b | 埋め込みモデル（Plamoで統一推奨）|
| `WHISPER_MODEL` | small | Whisper音声認識モデル |
| `WHISPER_LANGUAGE` | (auto) | Whisperへの言語ヒント（例: `ja`, `en`。未設定で自動検出）|
| `WHISPER_FORCE_LANGUAGE` | (unset) | 言語を完全に固定（例: `ja`）。歌唱で言語誤検出が起きる場合に有効 |
| `WHISPER_FORCE_MODE` | prefer | `always`=常に強制, `prefer`=自動検出を優先（高確度で別言語なら強制解除）, `off`=強制無効 |
| `WHISPER_FORCE_THRESHOLD` | 0.75 | 強制解除のしきい値（検出言語確度）。`prefer`モード時のみ使用 |
| `WHISPER_INITIAL_PROMPT` | (unset) | 文字起こしの初期プロンプト。例: 日本語歌詞を日本語で書き起こす旨 |
| `WHISPER_TEMPERATURE` | 0 | デコード温度（例: `0` または `0,0.2,0.4` のフォールバック列）|
| `WHISPER_BEAM_SIZE` | 5 | ビームサーチ幅（空なら未指定）|
| `WHISPER_CONDITION_ON_PREV` | 0 | 1で前文脈に条件付け（長尺会話向け）。歌詞での暴走防止には0推奨 |
| `ALLOW_TRANSLATION` | 0 | `1` かつ `WHISPER_TASK=translate` の時のみ翻訳を有効化。既定では原文そのままを文字起こし |
| `CAPTION_SNIPPET_CHARS` | 200 | Whisperのキャプション抜粋の最大文字数（全文は使わず抜粋のみ） |
| `AUDIO_INDEXER_SCRIPT` | audio_indexer_v2.py | 音声解析スクリプト |
| `ID_SCHEME` | rel8 | メディアID方式（rel8/rel16/abs8）|

#### 上級者向けENV（必要に応じて）
- `MEILI_INDEX`: Meilisearchのインデックス名（既定 `media`）。
- `MEILI_MASTER_KEY`: MeilisearchのAPIキー（既定 `masterKey`）。
- `QDRANT_COLLECTION`: Qdrantのコレクション名（既定 `media_vectors`）。
- `WHISPER_MAX_SECONDS`: Whisperで先頭N秒のみ解析。
- `WHISPER_OFFSET_SECONDS`: Whisperの開始オフセット秒。
- `AUDIO_TIMEOUT_MS`: 音声解析（Node側）のタイムアウトms（既定 180000〜480000）。
- `RERANK_MIN_RESULTS`/`RERANK_TOP`: Rerankを行う最小件数/上位K（既定 1 / 50）。
- `QDRANT_THRESHOLD`: セマンティック検索のしきい値（既定 0.25）。
- `SERVER_HOST`/`INDEX_HTML_PATH`: サーバーホスト/`index.html`パス。
- `PYTHON_VENV_PATH`: 仮想環境のPythonパス（例 `./venv/bin/python`）。

注記:
- 埋め込みは Plamo (`pfnet/plamo-embedding-1b`) でインデックス・検索の双方を統一する前提です。別モデルを使う場合は全体で揃えてください（次元不一致でQdrantエラーになります）。
- `.env(.example)` の `EXCLUDE_DIRS` は現状 `server.js` のスキャンでは未使用です（`MAX_DEPTH` は有効）。

### LM Studioセットアップ

1. [LM Studio](https://lmstudio.ai/)をダウンロード・インストール
2. LM Studio の Model Browser から視覚モデル（例：`qwen2.5-vl-7b`）をダウンロードして読み込み
3. settings > Serve on Local Networkをoff http://127.0.0.1:1234 
4. ポート1234でローカルサーバーを開始
5. 異なるモデルを使用する場合は`.env`の`LMSTUDIO_MODEL`を更新

補足:
- 画像解析: LM Studioのビジョンモデル（例: qwen2.5-vl-7b）を使用
- 音声解析(v2): Whisperで文字起こし＋特徴量解析。LM Studioはテキスト整形補助（任意）
- 文字起こし: 既定で原文言語のまま返します（日本語は日本語、英語は英語）。翻訳したい場合は `ALLOW_TRANSLATION=1` とし、合わせて `WHISPER_TASK=translate` を指定してください。
  - 歌唱で英語に化ける場合は、`WHISPER_FORCE_LANGUAGE=ja` を設定し、必要に応じて `WHISPER_INITIAL_PROMPT="日本語の歌詞をそのまま日本語で書き起こしてください。翻訳しない。"` を追加してください。モデルは `WHISPER_MODEL=medium` 以上を推奨。
- キャプション合成方針: まず Whisper の全文をキャプション先頭に採用。その後、音源が「楽曲」の場合のみ、楽曲解析（テンポ/ムード/楽器/カテゴリ等）の概要を追記します。会話の場合は楽曲解析やLMの整形は行いません。
- SVGロゴ: CairoSVGで白/黒背景にラスタライズした2枚をVLMに送信して誤判定（白黒反転やスピナー扱い）を抑制

## 使用方法

### 基本的なワークフロー

1. **メディアスキャン**: SCAN_PATHで設定したディレクトリ内のファイルを表示
2. **AI解析**: 「解析」ボタンを使用してキャプションとタグを生成
   - .glbの場合はサムネイルを生成してから解析スタート
3. **検索**: 
   - 「全文検索」高速なMeilisearch検索
   - 「セマンティック検索」AI類似検索を実行後、Rerankで並び替え

### サポートフォーマット

- **画像**: JPG, JPEG, PNG, GIF, SVG, WebP（SVGはラスタライズ後に解析）
- **動画**: MP4, MOV, WebM
- **音声**: MP3, WAV
- **3Dモデル**: GLB


## FFmpegのインストール

```bash
# macOS (Homebrew)
brew install ffmpeg

# Windows ffmpeg-master-latest-win64-gpl-shared.zip
https://github.com/BtbN/FFmpeg-Builds/releases

# 確認
ffmpeg -version
```

## 3Dサムネイル（GLB）

- GLBのサムネイルは、ブラウザ内の`<model-viewer>`でスナップショットを生成し、`.snapshots/` 配下に保存します。
- 保存形式はWebP（`*.webp`）を優先し、既存のPNG（`*.png`）がある場合はそのまま使用します。
- 一覧では、サムネイルが無い場合は🧊アイコンを表示し、モーダル表示や「解析」ボタン操作時に自動生成されます。
- 生成フロー（PC）
  - 解析ボタンはホバー時に表示（タッチ端末では常時表示）
  - サムネ未生成のときは「スナップショット生成中…」、続いて「解析中…」のローディングが表示され、解析モーダルが出たらローディングは消えます

## 謝辞

- 本プロジェクトは以下のOSSを元にしています。
  - media-gallery-viewer (by dai-motoki): https://github.com/dai-motoki/media-gallery-viewer
