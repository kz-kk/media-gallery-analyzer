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
| `EMB_MODEL` | all-MiniLM-L6-v2 | 埋め込みモデル（Plamo使用可）|
| `WHISPER_MODEL` | small | Whisper音声認識モデル |
| `AUDIO_INDEXER_SCRIPT` | audio_indexer_v2.py | 音声解析スクリプト |
| `ID_SCHEME` | rel8 | メディアID方式（rel8/rel16/abs8）|


### LM Studioセットアップ

1. [LM Studio](https://lmstudio.ai/)をダウンロード・インストール
2. LM Studio の Model Browser から視覚モデル（例：`qwen2.5-vl-7b`）をダウンロードして読み込み
3. settings > Serve on Local Networkをoff http://127.0.0.1:1234 
4. ポート1234でローカルサーバーを開始
5. 異なるモデルを使用する場合は`.env`の`LMSTUDIO_MODEL`を更新

補足:
- 画像解析: LM Studioのビジョンモデル（例: qwen2.5-vl-7b）を使用
- 音声解析(v2): Whisperで文字起こし＋特徴量解析。LM Studioはテキスト整形補助（任意）
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
- **動画**: MP4, MOV, AVI, MKV, WebM
- **音声**: MP3, WAV, OGG, FLAC, M4A
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
