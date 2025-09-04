#!/usr/bin/env python3
# coding: utf-8
"""
audio_indexer_v2.py
- librosaで音響特徴量を抽出
- 特徴量をLM Studio (テキストモデル)に送信してタグ生成
- Whisperで歌詞文字起こし
- Meilisearch/Qdrantに保存
"""
from pathlib import Path
import argparse
import json
import sys
import time
import os
import hashlib
from typing import Optional, Dict, List, Any
import mimetypes

# 依存ライブラリ
try:
    import requests
    from tqdm import tqdm
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from secure_logging import safe_debug_print, safe_json_dumps
    from audio_analyzer_v2 import analyze_audio_comprehensive
except Exception as e:
    print("依存ライブラリが不足しています:")
    print(str(e))
    raise

# 標準出力/標準エラーをUTF-8に（Windowsの文字化け対策）
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# .envファイル読み込み
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

# 設定
LMSTUDIO_URL = os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "qwen/qwen2.5-vl-7b")

MEILI_URL = os.getenv("MEILI_URL", "http://127.0.0.1:17700")
MEILI_API_KEY = os.getenv("MEILI_MASTER_KEY", "masterKey")
MEILI_INDEX = os.getenv("MEILI_INDEX", "media")

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:26333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "media_vectors")

# 事前取得を前提にPlamoを既定に（未取得/不可時は環境変数で上書き推奨）
EMB_MODEL = os.getenv("EMB_MODEL", "pfnet/plamo-embedding-1b")

DEFAULT_TIMEOUT = 300

# 対応する音声ファイル拡張子
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.wma', '.aac', '.opus', '.m4b'}

# ----------------- ユーティリティ -----------------
def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def compute_media_id(abs_path: str) -> int:
    """メディアID算出方式（デフォルト: rel8）。
    - rel8: SCAN_PATH相対のsha256先頭8hex → int（32bit）
    - rel16: SCAN_PATH相対のsha256先頭16hex → int（64bit）
    - abs8: absPathのsha256先頭8バイト → int（64bit）
    ID_SCHEME=rel8|rel16|abs8 で切替。
    """
    scheme = (os.getenv('ID_SCHEME') or 'rel8').lower().strip()
    if scheme in ('rel8', 'rel16'):
        scan = os.getenv('SCAN_PATH') or ''
        rel = abs_path
        try:
            if scan:
                rel = str(Path(abs_path).resolve().relative_to(Path(scan).resolve()))
        except Exception:
            rel = str(Path(abs_path).name)
        h = _sha256_hex(rel)
        if scheme == 'rel8':
            return int(h[:8], 16)
        else:
            return int(h[:16], 16)
    # default abs8
    hb = hashlib.sha256(abs_path.encode('utf-8')).digest()
    return int.from_bytes(hb[:8], byteorder='big')

# ----------------- LM Studio API -----------------
def analyze_features_with_lmstudio(features_desc: str, tags: List[str], 
                                  transcription: Optional[str] = None,
                                  model: str = LMSTUDIO_MODEL, 
                                  timeout: int = DEFAULT_TIMEOUT,
                                  features: Optional[dict] = None) -> Optional[dict]:
    """LM Studioで特徴量説明を分析してより高度なタグを生成"""
    
    safe_debug_print(f"[LM Studio] 特徴量をテキスト分析中...")
    
    # 解析上の制約ヒント（長調/短調・明暗）
    constraints = []
    if features:
        mode = features.get("mode")
        brightness = features.get("brightness_score")
        key = features.get("key")
        audio_type = features.get("audio_type")
        if audio_type == 'speech':
            constraints.append("これは音楽ではなく会話中心の音声。音楽用の用語（BPM/テンポ/キー/長調・短調など）は基本的に使わない")
        if mode:
            constraints.append(f"調性は{'長調' if mode=='major' else '短調'}（mode={mode}）")
        if key:
            constraints.append(f"推定キー: {key}")
        if isinstance(brightness, (int, float)):
            if brightness < 0.35:
                constraints.append("明暗は暗め（明るすぎる表現は禁止）")
            elif brightness < 0.55 and mode == 'minor':
                constraints.append("暗寄り（明るすぎる表現は避ける）")
            elif brightness > 0.65 and mode == 'major':
                constraints.append("明るめ（暗すぎる表現は避ける）")

    constraints_text = "\n".join([f"- {c}" for c in constraints]) if constraints else "- 特筆する制約なし"

    # プロンプト構築（矛盾禁止ルール込み）
    prompt = f"""以下の音声ファイルの解析結果から、適切なタグとより詳細な説明を日本語で生成してください。

音響特徴の説明:
{features_desc}

現在のタグ:
{', '.join(tags)}

解析上の制約:
{constraints_text}
"""
    
    if transcription:
        prompt += f"""
文字起こしテキスト:
{transcription[:500]}...
"""

    prompt += """
厳守事項:
- 上記の音響特徴と制約に矛盾する表現は禁止（例: 短調かつ暗めなのに「明るい」「キラキラ」「軽快」を多用しない）
- 明暗やムードは特徴量（brightness、mode）を最優先し、整合する語彙を用いる
- 断定しすぎず、必要なら保留的な言い回しを用いる
- 出力は次のJSON形式のみ。余分な文は出力しない

出力JSON形式:
{
  "improved_description": "音声の詳細な説明（ジャンル推定、雰囲気、使用シーンなどを含む）",
  "additional_tags": ["追加のタグ", "ジャンル", "雰囲気", "使用シーン", "楽器推定"],
  "genre_guess": "推定ジャンル",
  "mood": "雰囲気や感情（例: ダーク/明るい/哀愁/冷たい など）"
}
"""

    url = f"{LMSTUDIO_URL}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 500,
        "stream": False
    }
    
    try:
        r = requests.post(url, headers=headers, json=data, timeout=timeout)
        safe_debug_print(f"[LM Studio] Response status: {r.status_code}")
        r.raise_for_status()
        
        result = r.json()
        
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
        safe_debug_print(f"[LM Studio] 分析結果: {safe_json_dumps(parsed)}")
        
        return parsed
        
    except Exception as e:
        safe_debug_print(f"[LM Studio] エラー: {e}")
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
        "searchableAttributes": ["caption", "tags", "path", "transcription", "instruments", "category", "genre"],
        "displayedAttributes": ["id", "path", "caption", "tags", "model", "transcription", "features", "instruments", "category", "genre"],
        "filterableAttributes": ["tags", "model", "genre", "mood", "instruments", "category"],
        "sortableAttributes": ["path", "duration", "tempo"]
    }
    
    try:
        r = requests.patch(settings_url, json=settings_data, headers=headers)
        safe_debug_print(f"[Meilisearch] 設定更新: {r.status_code}")
    except Exception as e:
        safe_debug_print(f"[Meilisearch] 設定更新エラー: {e}")

def setup_qdrant(collection_name: str = QDRANT_COLLECTION, vector_size: int = 384):
    """Qdrant コレクションのセットアップ"""
    safe_debug_print(f"[Qdrant] セットアップ: name='{collection_name}', dim={vector_size}")
    
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
            safe_debug_print(f"[Qdrant] 作成: '{collection_name}' (dim={vector_size})")
        else:
            safe_debug_print(f"[Qdrant] 既存コレクションを使用: '{collection_name}'")
    except Exception as e:
        safe_debug_print(f"[Qdrant] セットアップエラー: {e}")

def _rel_path(p: str) -> str:
    scan = os.getenv('SCAN_PATH') or ''
    try:
        if scan:
            return str(Path(p).resolve().relative_to(Path(scan).resolve())).replace('\\', '/')
    except Exception:
        pass
    return str(Path(p).name)

def index_to_meilisearch(media_id: str, audio_analysis: Dict, lm_analysis: Optional[Dict] = None):
    """Meilisearch にドキュメントを登録"""
    safe_debug_print(f"[Meilisearch] ドキュメント登録: {audio_analysis['file_path']}")
    
    headers = {"Authorization": f"Bearer {MEILI_API_KEY}"}
    url = f"{MEILI_URL}/indexes/{MEILI_INDEX}/documents"
    
    # 基本情報
    doc = {
        "id": media_id,
        "path": _rel_path(audio_analysis["file_path"]),
        "caption": audio_analysis["description"],
        "tags": audio_analysis["tags"],
        "instruments": audio_analysis.get("instruments", []),
        "type": "audio",
        "indexed_at": int(time.time())
    }
    
    # 音響特徴量
    if audio_analysis.get("features"):
        doc.update({
            "duration": audio_analysis["features"].get("duration", 0),
            "tempo": audio_analysis["features"].get("tempo", 0),
            "features": audio_analysis["features"],
            "audio_type": audio_analysis["features"].get("audio_type", "")
        })
    
    # モデル名（会話ならWhisperと明示）
    try:
        audio_type = audio_analysis.get("features", {}).get("audio_type")
        doc["model"] = "Whisper" if audio_type == 'speech' else "RuleBased"
    except Exception:
        pass

    # 文字起こし
    if audio_analysis.get("transcription"):
        doc["transcription"] = audio_analysis["transcription"].get("text", "")
    
    # LM Studio分析結果
    if lm_analysis and (audio_analysis.get("features", {}).get("audio_type") != 'speech'):
        # Whisperの内容はLLMに渡していないため、captionは維持し、タグ/補助情報のみ追加
        additional_tags = lm_analysis.get("additional_tags", [])
        if isinstance(additional_tags, list):
            doc["tags"] = list(set(doc["tags"] + additional_tags))
        doc["genre"] = lm_analysis.get("genre_guess", "")
        doc["mood"] = lm_analysis.get("mood", "")
        # Try to normalize genre into category field
        try:
            from audio_analyzer_v2 import normalize_genre_label
            cat = normalize_genre_label(doc.get("genre", ""))
            if cat:
                doc["category"] = cat
                doc["tags"].append(cat)
        except Exception:
            pass
    # Heuristic category fallback
    if audio_analysis.get("category") and not doc.get("category"):
        doc["category"] = audio_analysis["category"]
    
    try:
        r = requests.post(url, json=[doc], headers=headers)
        safe_debug_print(f"[Meilisearch] 登録結果: {r.status_code}")
    except Exception as e:
        safe_debug_print(f"[Meilisearch] 登録エラー: {e}")

def index_to_qdrant(media_id: str, audio_analysis: Dict, embedding_model: Any, 
                   lm_analysis: Optional[Dict] = None, collection_name: str = QDRANT_COLLECTION):
    """Qdrant にベクトルを登録"""
    safe_debug_print(f"[Qdrant] ベクトル登録: {audio_analysis['file_path']}")
    
    client = QdrantClient(QDRANT_URL)
    
    # エンベディング用テキスト構築
    text_parts = [audio_analysis["description"]]
    text_parts.extend(audio_analysis["tags"])
    if audio_analysis.get("instruments"):
        text_parts.extend(audio_analysis["instruments"])
    
    if audio_analysis.get("transcription"):
        text_parts.append(audio_analysis["transcription"]["text"][:500])
    
    if lm_analysis and (audio_analysis.get("features", {}).get("audio_type") != 'speech'):
        text_parts.append(lm_analysis.get("improved_description", ""))
        text_parts.extend(lm_analysis.get("additional_tags", []))
    
    text = " ".join(text_parts)
    
    # エンベディング生成
    vec = embedding_model.encode(text)
    try:
        import numpy as _np
        norm = float(_np.linalg.norm(vec)) or 1.0
        vec = (vec / norm).tolist()
    except Exception:
        vec = vec.tolist() if hasattr(vec, 'tolist') else vec
    
    # ペイロード
    payload = {
        "media_id": media_id,
        "path": _rel_path(audio_analysis["file_path"]),
        "caption": audio_analysis["description"],
        "tags": audio_analysis["tags"],
        "instruments": audio_analysis.get("instruments", []),
        "type": "audio"
    }
    
    if audio_analysis.get("features"):
        payload.update({
            "duration": audio_analysis["features"].get("duration", 0),
            "tempo": audio_analysis["features"].get("tempo", 0),
            "audio_type": audio_analysis["features"].get("audio_type", "")
        })

    # モデル名（会話ならWhisper）
    try:
        audio_type = audio_analysis.get("features", {}).get("audio_type")
        payload["model"] = "Whisper" if audio_type == 'speech' else "RuleBased"
    except Exception:
        pass
    
    if lm_analysis:
        # captionは維持。LLM由来の追加タグ/メタのみ付与
        addt = lm_analysis.get("additional_tags", [])
        if isinstance(addt, list):
            payload["tags"] = list(set(payload["tags"] + addt))
        payload["genre"] = lm_analysis.get("genre_guess", "")
        payload["mood"] = lm_analysis.get("mood", "")
        try:
            from audio_analyzer_v2 import normalize_genre_label
            cat = normalize_genre_label(payload.get("genre", ""))
            if cat:
                payload["category"] = cat
                payload["tags"].append(cat)
        except Exception:
            pass
    # Heuristic category fallback
    if audio_analysis.get("category") and not payload.get("category"):
        payload["category"] = audio_analysis["category"]
    
    point = PointStruct(
        id=int(media_id),  # 整数として明示的にキャスト
        vector=vec,
        payload=payload
    )
    
    try:
        result = client.upsert(collection_name=collection_name, points=[point], wait=True)
        safe_debug_print(f"[Qdrant] upsert 完了: status={getattr(result, 'status', 'unknown')} id={media_id}")
    except Exception as e:
        safe_debug_print(f"[Qdrant] 登録エラー: {e}")

# ----------------- メイン処理 -----------------
def process_single_audio(audio_path: Path, embedding_model: Any,
                        use_lm_studio: bool = True, use_whisper: bool = True,
                        whisper_model: str = "base", collection_name: str = QDRANT_COLLECTION,
                        whisper_max_seconds: Optional[float] = None,
                        whisper_offset_seconds: float = 0.0):
    """単一の音声ファイルを処理"""
    safe_debug_print(f"\n{'='*60}")
    safe_debug_print(f"処理中: {audio_path}")
    
    # 1. 音声を総合的に解析
    audio_analysis = analyze_audio_comprehensive(
        audio_path, 
        use_whisper=use_whisper,
        whisper_model=whisper_model,
        whisper_max_seconds=whisper_max_seconds,
        whisper_offset_seconds=whisper_offset_seconds,
    )
    
    if not audio_analysis or not audio_analysis.get("features"):
        safe_debug_print(f"❌ 解析失敗: {audio_path}")
        return
    
    # 2. LM Studioで追加分析（オプション）
    lm_analysis = None
    # 会話主体の場合はLM整形をスキップ（音楽語彙への誘導を避ける）
    const_features = audio_analysis.get("features") or {}
    const_audio_type = const_features.get("audio_type")
    if use_lm_studio and const_audio_type not in ('speech', 'sfx'):
        # Whisper内容はLLMに渡さない（楽曲側の特徴に限定）
        lm_analysis = analyze_features_with_lmstudio(
            audio_analysis.get("music_description") or audio_analysis["description"],
            audio_analysis["tags"],
            None,
            features=audio_analysis.get("features")
        )
    
    # ID生成
    media_id = compute_media_id(str(audio_path))
    
    # 3. インデックス登録
    index_to_meilisearch(media_id, audio_analysis, lm_analysis)
    index_to_qdrant(media_id, audio_analysis, embedding_model, lm_analysis, collection_name)
    
    # 結果表示
    safe_debug_print(f"✅ 登録完了: {audio_path.name}")
    safe_debug_print(f"   説明: {audio_analysis['description'][:100]}...")
    safe_debug_print(f"   タグ: {', '.join(audio_analysis['tags'][:10])}")
    if audio_analysis.get("transcription"):
        safe_debug_print(f"   文字起こし: {audio_analysis['transcription']['text'][:50]}...")
    if lm_analysis:
        safe_debug_print(f"   ジャンル推定: {lm_analysis.get('genre_guess', 'N/A')}")
        safe_debug_print(f"   雰囲気: {lm_analysis.get('mood', 'N/A')}")
    
    # 結果を標準出力にも出力（JSON形式）
    # モデル名（会話ならWhisperと明示）
    atype = audio_analysis.get("features", {}).get("audio_type")
    model_name = "Whisper" if atype == 'speech' else ("SFX-Rules" if atype == 'sfx' else "RuleBased")
    result = {
        "success": True,
        "file_name": audio_path.name,
        "caption": audio_analysis["description"],
        "tags": list(set(audio_analysis["tags"] + lm_analysis.get("additional_tags", []))) if lm_analysis else audio_analysis["tags"],
        "media_id": media_id,
        "has_transcription": bool(audio_analysis.get("transcription")),
        "duration": audio_analysis["features"].get("duration", 0) if audio_analysis.get("features") else 0,
        "model": model_name
    }
    print(f"AUDIO_ANALYSIS_RESULT: {json.dumps(result, ensure_ascii=False)}")

def process_directory(media_dir: Path, embedding_model: Any, **kwargs):
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
            process_single_audio(audio_path, embedding_model, **kwargs)
        except Exception as e:
            safe_debug_print(f"エラー: {audio_path} - {e}")

class _EmbeddingWrapper:
    def __init__(self, st_model=None, plamo_model=None, plamo_tokenizer=None, is_plamo=False):
        self.st_model = st_model
        self.plamo_model = plamo_model
        self.plamo_tokenizer = plamo_tokenizer
        self.is_plamo = is_plamo
    def encode(self, text: str):
        if self.is_plamo:
            import torch
            with torch.inference_mode():
                # ドキュメント側は encode_document を使用（バッチ入力）
                vec = self.plamo_model.encode_document([text], self.plamo_tokenizer)
                import numpy as np
                if hasattr(vec, 'cpu'):
                    vec = vec.cpu().numpy()
                if isinstance(vec, list):
                    arr = np.array(vec)
                else:
                    arr = vec
                # 先頭要素（1件分）を取り出し1次元に
                if getattr(arr, 'ndim', 1) > 1:
                    arr = arr[0]
                return arr
        else:
            return self.st_model.encode(text)

def main():
    parser = argparse.ArgumentParser(description="音声ファイルをlibrosa/Whisper/LM Studioで解析")
    parser.add_argument("--audio", type=str, help="単一音声ファイルのパス")
    parser.add_argument("--media_dir", type=str, help="音声ディレクトリのパス（再帰処理）")
    parser.add_argument("--no-lm-studio", action="store_true", help="LM Studioを使用しない")
    parser.add_argument("--no-whisper", action="store_true", help="Whisperを使用しない")
    parser.add_argument("--whisper-model", type=str, default="base",
                      choices=["tiny", "base", "small", "medium", "large"],
                      help="Whisperモデルサイズ (デフォルト: base)")
    parser.add_argument("--embedding_model", type=str, default=EMB_MODEL,
                      help=f"エンベディングモデル (デフォルト: {EMB_MODEL})")
    parser.add_argument("--whisper-max-seconds", type=float, default=float(os.getenv('WHISPER_MAX_SECONDS') or 0.0),
                      help="Whisperで文字起こしする最大秒数（0で無制限）")
    parser.add_argument("--whisper-offset-seconds", type=float, default=float(os.getenv('WHISPER_OFFSET_SECONDS') or 0.0),
                      help="Whisperで部分文字起こしを行う開始オフセット秒")
    
    args = parser.parse_args()
    
    if not args.audio and not args.media_dir:
        parser.print_help()
        sys.exit(1)
    
    # セットアップ
    setup_meilisearch()
    
    # エンベディングモデル選択（ENV/引数の EMB_MODEL を尊重）
    embedding_model = None
    vector_size = 384
    collection_name = QDRANT_COLLECTION
    emb_name = args.embedding_model or EMB_MODEL
    try:
        if 'plamo' in (emb_name or '').lower():
            from transformers import AutoTokenizer, AutoModel
            import torch
            # Hugging Faceキャッシュから読み込み（自動的にキャッシュされる）
            safe_debug_print(f"モデルをロード中: {emb_name}")
            tok = AutoTokenizer.from_pretrained(emb_name, trust_remote_code=True)
            mdl = AutoModel.from_pretrained(emb_name, trust_remote_code=True)
            with torch.inference_mode():
                test_vec = mdl.encode_document(["test"], tok)
                if hasattr(test_vec, 'shape'):
                    vector_size = int(getattr(test_vec, 'shape')[-1])
                else:
                    vector_size = len(test_vec[0])
            embedding_model = _EmbeddingWrapper(plamo_model=mdl, plamo_tokenizer=tok, is_plamo=True)
        else:
            # sentence-transformers 系
            st = SentenceTransformer(emb_name)
            # 次元数取得
            vec = st.encode("test")
            try:
                import numpy as _np
                vector_size = int((_np.asarray(vec)).shape[-1])
            except Exception:
                vector_size = len(vec) if hasattr(vec, '__len__') else 384
            embedding_model = _EmbeddingWrapper(st_model=st, is_plamo=False)
    except Exception as e:
        safe_debug_print(f"Embedding model load failed ({emb_name}): {e}")
        raise

    setup_qdrant(collection_name=collection_name, vector_size=vector_size)
    
    # 処理オプション
    kwargs = {
        "use_lm_studio": not args.no_lm_studio,
        "use_whisper": not args.no_whisper,
        "whisper_model": args.whisper_model,
        "whisper_max_seconds": (args.whisper_max_seconds if args.whisper_max_seconds and args.whisper_max_seconds > 0 else None),
        "whisper_offset_seconds": (args.whisper_offset_seconds or 0.0),
    }
    
    # 処理実行
    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            safe_debug_print(f"ファイルが見つかりません: {audio_path}")
            sys.exit(1)
        process_single_audio(audio_path, embedding_model, collection_name=collection_name, **kwargs)
    
    elif args.media_dir:
        media_dir = Path(args.media_dir)
        if not media_dir.exists():
            safe_debug_print(f"ディレクトリが見つかりません: {media_dir}")
            sys.exit(1)
        process_directory(media_dir, embedding_model, collection_name=collection_name, **kwargs)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    try:
        main()
    finally:
        # プロセス終了時のクリーンアップ
        import gc
        gc.collect()
