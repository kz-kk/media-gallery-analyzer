#!/usr/bin/env python3
# coding: utf-8
"""
audio_indexer.py
- LM Studio (OpenAI互換API) に音声を送り JSON(caption/tags) を取得し、
  Meilisearch に全文ドキュメント登録、Qdrant にベクトル登録する
- 単一ファイル (--audio) / ディレクトリ再帰 (--media_dir) 対応
- Qwen2-Audio-7B-Instruct モデル使用
"""
from pathlib import Path
import argparse
import base64
import json
import sys
import time
import os
import hashlib
from typing import Optional, Dict
import mimetypes
import subprocess
import tempfile
import wave
import struct

# 依存: requests mutagen qdrant-client sentence-transformers tqdm numpy pydub
try:
    import requests
    from mutagen import FileType as MutagenFileType
    from tqdm import tqdm
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from secure_logging import safe_debug_print, sanitize_dict, safe_json_dumps
    from pydub import AudioSegment
    from pydub.utils import mediainfo
except Exception as e:
    print("依存ライブラリが不足しています。以下をインストールしてください:")
    print("python -m pip install qdrant-client sentence-transformers requests mutagen tqdm numpy pydub")
    raise

# ----------------- .envファイル読み込み -----------------
def load_env():
    """Load .env file variables"""
    env_file = Path('.env')
    if env_file.exists():
        with env_file.open('r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# .envファイルを読み込み
load_env()

# ----------------- 設定（環境変数 or CLI 引数で上書き可） -----------------
LMSTUDIO_URL = os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "qwen2-audio-7b-instruct")  # 音声モデル

MEILI_URL = os.getenv("MEILI_URL", "http://127.0.0.1:17700")
MEILI_API_KEY = os.getenv("MEILI_MASTER_KEY", "masterKey")
MEILI_INDEX = os.getenv("MEILI_INDEX", "media")

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:26333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "media_vectors")

EMB_MODEL = os.getenv("EMB_MODEL", "all-MiniLM-L6-v2")

DEFAULT_TIMEOUT = 300
DEFAULT_CHUNK_DURATION = int(os.getenv('AUDIO_CHUNK_SEC', '30'))  # 30秒ごとに分割（ENVで上書き可）
MAX_CHUNKS = int(os.getenv('AUDIO_MAX_CHUNKS', '5'))  # 最大チャンク数（ENVで上書き可）

PROMPT = (
    "この音声を聴いて、JSON形式で回答してください。\n"
    "JSON以外は出力しないでください。\n\n"
    "{\"caption\": \"音声の説明（内容、ジャンル、雰囲気を含む）\", \"tags\": [\"ジャンル\", \"楽器\", \"雰囲気\", \"テンポ\", \"その他\"]}\n\n"
    "例：\n"
    "{\"caption\": \"アコースティックギターによる穏やかなメロディー\", \"tags\": [\"音楽\", \"ギター\", \"アコースティック\", \"穏やか\", \"インストゥルメンタル\", \"リラックス\"]}"
)

# 対応する音声ファイル拡張子
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.wma', '.aac'}

# ----------------- ユーティリティ -----------------
def path_sha256_hex(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()

def get_audio_metadata(path: Path) -> Dict:
    """音声ファイルのメタデータを取得（WAVはwaveモジュールでフォールバック）"""
    try:
        audio = MutagenFileType(path)
        metadata: Dict = {"duration": 0, "bitrate": 0, "sample_rate": 0}

        if audio is not None and getattr(audio, 'info', None) is not None:
            metadata.update({
                "duration": float(getattr(audio.info, 'length', 0) or 0),
                "bitrate": int(getattr(audio.info, 'bitrate', 0) or 0),
                "sample_rate": int(getattr(audio.info, 'sample_rate', 0) or 0),
                "channels": int(getattr(audio.info, 'channels', 0) or 0),
            })

            # タイトル、アーティストなどのタグ情報も取得
            if hasattr(audio, 'tags') and audio.tags:
                try:
                    metadata['title'] = audio.tags.get('title', [None])[0] if 'title' in audio.tags else None
                    metadata['artist'] = audio.tags.get('artist', [None])[0] if 'artist' in audio.tags else None
                    metadata['album'] = audio.tags.get('album', [None])[0] if 'album' in audio.tags else None
                except Exception:
                    pass

        # WAVなどでmutagenが長さを返せない場合はwaveで推定
        ext = path.suffix.lower()
        if (not metadata.get('duration')) and ext == '.wav':
            try:
                with wave.open(str(path), 'rb') as wf:
                    n_frames = wf.getnframes()
                    framerate = wf.getframerate() or 1
                    channels = wf.getnchannels() or 0
                    metadata['duration'] = float(n_frames) / float(framerate)
                    metadata['sample_rate'] = int(framerate)
                    metadata['channels'] = int(channels)
            except Exception as we:
                safe_debug_print(f"WAVメタデータ取得エラー: {we}")

        return metadata
    except Exception as e:
        safe_debug_print(f"メタデータ取得エラー: {e}")
        return {"duration": 0, "bitrate": 0, "sample_rate": 0}

def audio_to_base64(path: Path) -> str:
    """音声ファイルをBase64エンコード"""
    try:
        with open(path, 'rb') as f:
            audio_bytes = f.read()
        return base64.b64encode(audio_bytes).decode('utf-8')
    except Exception as e:
        safe_debug_print(f"音声ファイル読み込みエラー: {e}")
        raise

def split_audio(audio_path: Path, chunk_duration: int = DEFAULT_CHUNK_DURATION, 
                max_chunks: int = MAX_CHUNKS) -> list:
    """長い音声ファイルを分割"""
    safe_debug_print(f"音声ファイルを分割中: {audio_path}")
    
    try:
        # pydubで音声を読み込み
        audio = AudioSegment.from_file(str(audio_path))
        duration = len(audio) / 1000  # ミリ秒を秒に変換
        safe_debug_print(f"音声の長さ: {duration:.1f}秒")
        
        # 短い音声はそのまま返す
        if duration <= chunk_duration:
            safe_debug_print("音声が短いため分割しません")
            return [audio_path]
        
        # 分割する
        chunks = []
        temp_dir = tempfile.mkdtemp()
        chunk_count = min(int(duration / chunk_duration) + 1, max_chunks)
        
        for i in range(chunk_count):
            start = i * chunk_duration * 1000  # 秒をミリ秒に
            end = min(start + chunk_duration * 1000, len(audio))
            
            # 最後のチャンクに達したら終了
            if i >= max_chunks - 1:
                end = len(audio)
            
            chunk = audio[start:end]
            chunk_path = Path(temp_dir) / f"chunk_{i}.mp3"
            chunk.export(str(chunk_path), format="mp3")
            chunks.append(chunk_path)
            safe_debug_print(f"チャンク{i+1}: {start/1000:.1f}秒 - {end/1000:.1f}秒")
            
            # 最大チャンク数に達したら終了
            if i >= max_chunks - 1:
                break
        
        return chunks
        
    except Exception as e:
        safe_debug_print(f"音声分割エラー: {e}")
        # エラーの場合は元のファイルを返す
        return [audio_path]

# ----------------- LM Studio API -----------------
def analyze_audio_lmstudio(audio_path: Path, model: str = LMSTUDIO_MODEL, 
                          timeout: int = DEFAULT_TIMEOUT) -> Optional[dict]:
    """LM Studio (OpenAI互換API) に音声を送って解析"""
    
    safe_debug_print(f"[LM Studio] 音声解析中: {audio_path}")
    
    # 音声をBase64エンコード
    audio_base64 = audio_to_base64(audio_path)
    
    # MIMEタイプを取得
    mime_type, _ = mimetypes.guess_type(str(audio_path))
    if not mime_type:
        mime_type = "audio/mpeg"  # デフォルト
    
    # メッセージ構築（Qwen2-Audio形式）
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": f"data:{mime_type};base64,{audio_base64}"
                    }
                }
            ]
        }
    ]
    
    url = f"{LMSTUDIO_URL}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 500,
        "stream": False
    }
    
    sanitized_data = sanitize_dict(data)
    safe_debug_print(f"[LM Studio] Request URL: {url}")
    safe_debug_print(f"[LM Studio] Request data: {safe_json_dumps(sanitized_data)}")
    
    try:
        r = requests.post(url, headers=headers, json=data, timeout=timeout)
        safe_debug_print(f"[LM Studio] Response status: {r.status_code}")
        r.raise_for_status()
        
        result = r.json()
        safe_debug_print(f"[LM Studio] Response: {safe_json_dumps(result)}")
        
        if "choices" not in result or not result["choices"]:
            safe_debug_print(f"[LM Studio] No choices in response")
            return None
            
        content = result["choices"][0]["message"]["content"]
        
        # JSON部分を抽出
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        
        parsed = json.loads(content.strip())
        safe_debug_print(f"[LM Studio] 解析結果: {safe_json_dumps(parsed)}")
        
        return parsed
        
    except requests.exceptions.RequestException as e:
        safe_debug_print(f"[LM Studio] APIエラー: {e}")
        return None
    except json.JSONDecodeError as e:
        safe_debug_print(f"[LM Studio] JSON解析エラー: {e}")
        safe_debug_print(f"[LM Studio] 受信内容: {content if 'content' in locals() else 'N/A'}")
        return None
    except Exception as e:
        safe_debug_print(f"[LM Studio] 予期しないエラー: {e}")
        return None

# ----------------- Meilisearch & Qdrant -----------------
def setup_meilisearch(index_name: str = MEILI_INDEX):
    """Meilisearch インデックスのセットアップ"""
    safe_debug_print(f"[Meilisearch] インデックス '{index_name}' をセットアップ中...")
    
    headers = {"Authorization": f"Bearer {MEILI_API_KEY}"}
    
    # インデックス作成
    create_url = f"{MEILI_URL}/indexes"
    create_data = {"uid": index_name, "primaryKey": "id"}
    
    try:
        r = requests.post(create_url, json=create_data, headers=headers)
        if r.status_code in [200, 201]:
            safe_debug_print(f"[Meilisearch] インデックス '{index_name}' を作成しました")
        elif r.status_code == 202:
            safe_debug_print(f"[Meilisearch] インデックス '{index_name}' の作成タスクを受け付けました")
        else:
            safe_debug_print(f"[Meilisearch] インデックス作成: {r.status_code} - {r.text}")
    except Exception as e:
        safe_debug_print(f"[Meilisearch] インデックス作成エラー: {e}")
    
    # 設定更新
    settings_url = f"{MEILI_URL}/indexes/{index_name}/settings"
    settings_data = {
        "searchableAttributes": ["caption", "tags", "path"],
        "displayedAttributes": ["id", "path", "caption", "tags", "model"],
        "filterableAttributes": ["tags", "model"],
        "sortableAttributes": ["path"]
    }
    
    try:
        r = requests.patch(settings_url, json=settings_data, headers=headers)
        safe_debug_print(f"[Meilisearch] 設定更新: {r.status_code}")
    except Exception as e:
        safe_debug_print(f"[Meilisearch] 設定更新エラー: {e}")

def setup_qdrant(collection_name: str = QDRANT_COLLECTION, vector_size: int = 384):
    """Qdrant コレクションのセットアップ"""
    safe_debug_print(f"[Qdrant] コレクション '{collection_name}' をセットアップ中...")
    
    client = QdrantClient(QDRANT_URL)
    
    try:
        # コレクション情報を取得
        collections = client.get_collections()
        exists = any(c.name == collection_name for c in collections.collections)
        
        if not exists:
            # コレクション作成
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            safe_debug_print(f"[Qdrant] コレクション '{collection_name}' を作成しました")
        else:
            safe_debug_print(f"[Qdrant] コレクション '{collection_name}' は既に存在します")
    except Exception as e:
        safe_debug_print(f"[Qdrant] セットアップエラー: {e}")

def index_to_meilisearch(media_id: str, path: str, caption: str, tags: list, 
                        model: str = LMSTUDIO_MODEL, metadata: dict = None):
    """Meilisearch にドキュメントを登録"""
    safe_debug_print(f"[Meilisearch] ドキュメント登録: {path}")
    
    headers = {"Authorization": f"Bearer {MEILI_API_KEY}"}
    url = f"{MEILI_URL}/indexes/{MEILI_INDEX}/documents"
    
    doc = {
        "id": media_id,
        "path": path,
        "caption": caption,
        "tags": tags,
        "model": model,
        "type": "audio",
        "indexed_at": int(time.time())
    }
    
    # メタデータを追加
    if metadata:
        doc.update({
            "duration": metadata.get("duration", 0),
            "bitrate": metadata.get("bitrate", 0),
            "sample_rate": metadata.get("sample_rate", 0),
            "title": metadata.get("title"),
            "artist": metadata.get("artist"),
            "album": metadata.get("album")
        })
    
    try:
        r = requests.post(url, json=[doc], headers=headers)
        safe_debug_print(f"[Meilisearch] 登録結果: {r.status_code}")
    except Exception as e:
        safe_debug_print(f"[Meilisearch] 登録エラー: {e}")

def index_to_qdrant(media_id: str, path: str, caption: str, tags: list, 
                   embedding_model: SentenceTransformer, model: str = LMSTUDIO_MODEL,
                   metadata: dict = None):
    """Qdrant にベクトルを登録"""
    safe_debug_print(f"[Qdrant] ベクトル登録: {path}")
    
    client = QdrantClient(QDRANT_URL)
    
    # キャプションとタグを結合してテキスト化
    text = f"{caption} {' '.join(tags)}"
    
    # エンベディング生成
    vector = embedding_model.encode(text).tolist()
    
    payload = {
        "media_id": media_id,
        "path": path,
        "caption": caption,
        "tags": tags,
        "model": model,
        "type": "audio"
    }
    
    # メタデータを追加
    if metadata:
        payload.update({
            "duration": metadata.get("duration", 0),
            "bitrate": metadata.get("bitrate", 0),
            "sample_rate": metadata.get("sample_rate", 0),
            "title": metadata.get("title"),
            "artist": metadata.get("artist"),
            "album": metadata.get("album")
        })
    
    point = PointStruct(
        id=media_id,
        vector=vector,
        payload=payload
    )
    
    try:
        client.upsert(collection_name=QDRANT_COLLECTION, points=[point])
        safe_debug_print(f"[Qdrant] ベクトル登録完了")
    except Exception as e:
        safe_debug_print(f"[Qdrant] 登録エラー: {e}")

# ----------------- メイン処理 -----------------
def process_single_audio(audio_path: Path, embedding_model: SentenceTransformer,
                        model: str = LMSTUDIO_MODEL, timeout: int = DEFAULT_TIMEOUT,
                        chunk_sec: int = DEFAULT_CHUNK_DURATION, max_chunks: int = MAX_CHUNKS,
                        use_lm_studio: bool = True):
    """単一の音声ファイルを処理"""
    safe_debug_print(f"\n{'='*60}")
    safe_debug_print(f"処理中: {audio_path}")
    
    # メタデータ取得
    metadata = get_audio_metadata(audio_path)
    safe_debug_print(f"メタデータ: {safe_json_dumps(metadata)}")
    
    # 音声を分割（長い場合）
    chunks = split_audio(audio_path, chunk_duration=chunk_sec, max_chunks=max_chunks)
    safe_debug_print(f"チャンク数: {len(chunks)}")
    
    # 各チャンクを解析
    all_captions = []
    all_tags = set()
    chunk_results = []
    
    for i, chunk_path in enumerate(chunks):
        safe_debug_print(f"\nチャンク {i+1}/{len(chunks)} を解析中...")
        
        analysis = None
        if use_lm_studio:
            analysis = analyze_audio_lmstudio(chunk_path, model=model, timeout=timeout)
        
        if analysis:
            caption = analysis.get("caption", "")
            tags = analysis.get("tags", [])
            
            if caption:
                all_captions.append(f"セクション{i+1}: {caption}")
            if tags:
                all_tags.update(tags)
            
            chunk_results.append(analysis)
        
        # 一時ファイルの場合は削除
        if chunk_path != audio_path:
            try:
                chunk_path.unlink()
            except:
                pass
    
    if not chunk_results:
        # Fallback: LM Studioが使えない/失敗でもメタデータから簡易キャプション・タグを生成して継続
        safe_debug_print(f"⚠️ 解析結果なし。メタデータのみでフォールバック: {audio_path}")
        duration = metadata.get('duration', 0)
        duration_str = f"{duration:.1f}秒" if duration > 0 else "不明"
        combined_caption = f"音声ファイル ({duration_str})"
        combined_tags = []
        if duration > 0:
            if duration < 10:
                combined_tags.append("短い")
            elif duration < 60:
                combined_tags.append("中程度")
            else:
                combined_tags.append("長い")
            combined_tags.append(f"{int(duration)}秒")
        if metadata.get("title"): combined_tags.append(metadata["title"])
        if metadata.get("artist"): combined_tags.append(metadata["artist"])
        if metadata.get("album"): combined_tags.append(metadata["album"])
        combined_tags.extend(["音声", "audio", duration_str])
        media_id = path_sha256_hex(str(audio_path))
        # 軽量登録（Meiliのみ）。Qdrantはベクトル不要のため省略可
        try:
            index_to_meilisearch(media_id, str(audio_path), combined_caption, combined_tags, model, metadata)
        except Exception as e:
            safe_debug_print(f"Meilisearch登録失敗(フォールバック): {e}")
        # 結果をstdoutに出力して呼び出し元へ伝達
        result = {
            "success": True,
            "file_name": audio_path.name,
            "caption": combined_caption,
            "tags": combined_tags,
            "media_id": media_id,
            "has_transcription": False,
            "duration": int(duration) if duration else 0,
            "model": model
        }
        print(f"AUDIO_ANALYSIS_RESULT: {json.dumps(result, ensure_ascii=False)}")
        return
    
    # 結果を統合
    duration_str = f"{metadata.get('duration', 0):.1f}秒" if metadata.get('duration', 0) > 0 else "不明"
    combined_caption = f"音声ファイル ({duration_str}): " + " / ".join(all_captions[:3]) if all_captions else f"音声ファイル ({duration_str})"
    combined_tags = list(all_tags)
    
    # 音声関連のタグを追加
    if metadata.get("duration", 0) > 0:
        duration = metadata["duration"]
        if duration < 10:
            combined_tags.append("短い")
        elif duration < 60:
            combined_tags.append("中程度")
        else:
            combined_tags.append("長い")
        combined_tags.append(f"{int(duration)}秒")
    
    # メタデータからタグを追加
    if metadata.get("title"):
        combined_tags.append(metadata["title"])
    if metadata.get("artist"):
        combined_tags.append(metadata["artist"])
    if metadata.get("album"):
        combined_tags.append(metadata["album"])
    
    combined_tags.extend(["音声", "audio", duration_str])
    
    # ID生成
    media_id = path_sha256_hex(str(audio_path))
    
    # インデックス登録
    index_to_meilisearch(media_id, str(audio_path), combined_caption, combined_tags, model, metadata)
    index_to_qdrant(media_id, str(audio_path), combined_caption, combined_tags, embedding_model, model, metadata)
    
    safe_debug_print(f"✅ 登録完了: {audio_path.name}")
    safe_debug_print(f"   キャプション: {combined_caption}")
    safe_debug_print(f"   タグ: {', '.join(combined_tags[:10])}")  # 最初の10個のタグを表示
    safe_debug_print(f"   チャンク数: {len(chunks)}")
    
    # 結果を標準出力にも出力（JSON形式）
    result = {
        "success": True,
        "file_name": audio_path.name,
        "caption": combined_caption,
        "tags": combined_tags,
        "audio_metadata": metadata,
        "media_id": media_id,
        "analyzed_chunks": len(chunks)
    }
    print(f"AUDIO_ANALYSIS_RESULT: {json.dumps(result, ensure_ascii=False)}")

def process_directory(media_dir: Path, embedding_model: SentenceTransformer,
                     model: str = LMSTUDIO_MODEL, timeout: int = DEFAULT_TIMEOUT):
    """ディレクトリ内の音声ファイルを再帰的に処理"""
    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(media_dir.rglob(f"*{ext}"))
        audio_files.extend(media_dir.rglob(f"*{ext.upper()}"))
    
    safe_debug_print(f"見つかった音声ファイル: {len(audio_files)}個")
    
    for audio_path in tqdm(audio_files, desc="音声処理中"):
        if not audio_path.is_file():
            continue
        
        try:
            process_single_audio(audio_path, embedding_model, model=model, timeout=timeout)
        except Exception as e:
            safe_debug_print(f"エラー: {audio_path} - {e}")

def main():
    parser = argparse.ArgumentParser(description="音声ファイルをLM Studioで解析し、Meilisearch/Qdrantに登録")
    parser.add_argument("--audio", type=str, help="単一音声ファイルのパス")
    parser.add_argument("--media_dir", type=str, help="音声ディレクトリのパス（再帰処理）")
    parser.add_argument("--model", type=str, default=LMSTUDIO_MODEL,
                      help=f"LM Studioモデル名 (デフォルト: {LMSTUDIO_MODEL})")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                      help=f"APIタイムアウト秒数 (デフォルト: {DEFAULT_TIMEOUT})")
    parser.add_argument("--embedding_model", type=str, default=EMB_MODEL,
                      help=f"エンベディングモデル (デフォルト: {EMB_MODEL})")
    parser.add_argument("--chunk-sec", type=int, default=DEFAULT_CHUNK_DURATION,
                      help=f"1チャンクの秒数（デフォルト: {DEFAULT_CHUNK_DURATION}）")
    parser.add_argument("--max-chunks", type=int, default=MAX_CHUNKS,
                      help=f"最大チャンク数（デフォルト: {MAX_CHUNKS}）")
    parser.add_argument("--no-lm-studio", action="store_true",
                      help="LM Studioを使用せず、メタデータのみで処理（フォールバック）")
    
    args = parser.parse_args()
    
    if not args.audio and not args.media_dir:
        parser.print_help()
        sys.exit(1)
    
    # セットアップ
    setup_meilisearch()
    embedding_model = SentenceTransformer(args.embedding_model, trust_remote_code=True)
    vector_size = embedding_model.get_sentence_embedding_dimension()
    setup_qdrant(vector_size=vector_size)

    # モデル名から自動的にLM Studio無効化（音声非対応モデルを回避）
    auto_disable_lm = False
    try:
        m = (args.model or "").lower()
        # 音声対応のヒューリスティクス: "audio" を含まない、または "vl" を含むモデルは無効化
        if ("audio" not in m) or ("vl" in m):
            auto_disable_lm = True
            safe_debug_print(f"[LM Studio] モデル '{args.model}' は音声非対応の可能性が高いためLM Studio呼び出しを無効化します")
    except Exception:
        pass

    use_lm = not args.no_lm_studio and not auto_disable_lm

    # 処理実行
    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            safe_debug_print(f"ファイルが見つかりません: {audio_path}")
            sys.exit(1)
        process_single_audio(audio_path, embedding_model, model=args.model, timeout=args.timeout,
                            chunk_sec=args.chunk_sec, max_chunks=args.max_chunks, use_lm_studio=use_lm)
    
    elif args.media_dir:
        media_dir = Path(args.media_dir)
        if not media_dir.exists():
            safe_debug_print(f"ディレクトリが見つかりません: {media_dir}")
            sys.exit(1)
        # ディレクトリ処理では引数のチャンク設定を渡す
        def _proc_dir(media_dir: Path):
            audio_files = []
            for ext in AUDIO_EXTENSIONS:
                audio_files.extend(media_dir.rglob(f"*{ext}"))
                audio_files.extend(media_dir.rglob(f"*{ext.upper()}"))
            for ap in tqdm(audio_files, desc="音声処理中"):
                try:
                    process_single_audio(ap, embedding_model, model=args.model, timeout=args.timeout,
                                        chunk_sec=args.chunk_sec, max_chunks=args.max_chunks, use_lm_studio=use_lm)
                except Exception as e:
                    safe_debug_print(f"エラー: {ap} - {e}")
        _proc_dir(media_dir)

if __name__ == "__main__":
    main()
