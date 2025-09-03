#!/usr/bin/env python3
# coding: utf-8
"""
media_indexer_lmstudio.py
- LM Studio (OpenAI互換API) に画像を送り JSON(caption/tags) を取得し、
  Meilisearch に全文ドキュメント登録、Qdrant にベクトル登録する
- 単一ファイル (--image) / ディレクトリ再帰 (--media_dir) 対応
"""
from pathlib import Path
import argparse
import base64
import json
import sys
import tempfile
import time
import os
import sys
import hashlib
from typing import Optional, List

# 依存: requests pillow qdrant-client sentence-transformers tqdm numpy
try:
    import requests
    from PIL import Image, UnidentifiedImageError, ImageFile, ImageSequence
    # ★ PNG/巨大画像/途中切れ対策（Pillowの直後で設定）
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    from tqdm import tqdm
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
    from sentence_transformers import SentenceTransformer
    import numpy as np  # ベクトル型変換に使用
    from secure_logging import safe_debug_print, sanitize_dict, safe_json_dumps
    import shutil
    import subprocess
except Exception as e:
    print("依存ライブラリが不足しています。以下をインストールしてください:")
    print("python -m pip install qdrant-client sentence-transformers requests pillow tqdm numpy")
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
try:
    # Ensure UTF-8 stdout/stderr to avoid mojibake on Windows when captured by Node
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

LMSTUDIO_URL = os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "qwen2.5-vl-7b")

MEILI_URL = os.getenv("MEILI_URL", "http://127.0.0.1:17700")
MEILI_API_KEY = os.getenv("MEILI_MASTER_KEY", "masterKey")
MEILI_INDEX = os.getenv("MEILI_INDEX", "media")

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:26333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "media_vectors")

EMB_MODEL = os.getenv("EMB_MODEL", "all-MiniLM-L6-v2")

DEFAULT_RESIZE = 1024
DEFAULT_QUALITY = 85
DEFAULT_TIMEOUT = 300

PROMPT = (
    "次の画像について、日本語で詳細に記述し、必ずJSONのみを返してください。\n"
    "出力形式:\n"
    "{\n"
    "  \"caption\": \"60〜140文字で具体的な説明（被写体/構図/背景/光や色/雰囲気/用途を含め、\n"
    "                 ‘美しい/きれい’など抽象表現に偏らず、事実に基づく描写）\",\n"
    "  \"tags\": [\"被写体\", \"場所/シーン\", \"スタイル/画風\", \"色/配色\", \"ライティング\", \"雰囲気\", \"小物/要素\", \"用途\", \"視点/構図\"]\n"
    "}\n\n"
    "制約:\n"
    "- JSON以外は出力しない（説明文や注釈を混ぜない）\n"
    "- ‘美しい/きれい/素敵’等の抽象語だけに頼らず、具体的な名詞/形容で描写\n"
    "- 被写体が少ない場合も背景や光の向き、質感、季節・時間帯などを補足\n"
    "- イラスト/写真/3D/CG等のタイプが分かる場合はcaptionに含める\n"
    "- 可能なら用途（壁紙/広告/商品写真等）や構図（俯瞰/クローズアップ等）も含める\n\n"
    "悪い例: {\"caption\": \"夜空の写真\", \"tags\": [\"夜\"]}\n"
    "良い例: {\"caption\": \"冬の山岳地帯で撮影された写真。緑と紫のオーロラが弧を描き、\n"
    "            雪原の地平線と松のシルエットが手前に重なる。長時間露光で星が微かに流れ、\n"
    "            冷たい月光が淡く照らす。\",\n"
    "          \"tags\": [\"オーロラ\", \"星空\", \"夜景\", \"雪原\", \"松の木\", \"冬\", \"長時間露光\", \"寒色\", \"写真\", \"風景\"]}"
)

# ----------------- ユーティリティ -----------------
def path_sha256_hex(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()

def compute_media_id_from_rel(rel: str) -> int:
    """ID_SCHEME に応じて media_id を算出（デフォルト: rel8）。
    - rel8: sha256先頭8hex → int（32bit）
    - rel16: sha256先頭16hex → int（64bit）
    画像は相対パス基準。
    """
    scheme = (os.getenv('ID_SCHEME') or 'rel8').lower().strip()
    h = path_sha256_hex(rel)
    if scheme == 'rel16':
        return int(h[:16], 16)
    return int(h[:8], 16)

def convert_and_resize_to_jpeg(path: Path, width: int = 1024, quality: int = 85) -> Path:
    """画像をJPEGに変換・リサイズして一時ファイルを返す"""
    try:
        # SVG はここで弾かず、別経路でラスタライズ対応
        if str(path).lower().endswith('.svg'):
            raise ValueError("use rasterize_svg_to_png for svg")
            
        with Image.open(path) as img:
            # RGBに変換
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # リサイズ
            if img.width > width or img.height > width:
                img.thumbnail((width, width), Image.Resampling.LANCZOS)
            
            # 一時ファイルにJPEGで保存
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            img.save(tmp_file.name, 'JPEG', quality=quality)
            return Path(tmp_file.name)
            
    except (UnidentifiedImageError, OSError, IOError, ValueError) as e:
        # 画像ファイルとして認識できない、または破損している場合
        print(f"[WARNING] {path} is not a valid image or corrupted: {e}")
        raise ValueError(f"Cannot process image: {path}")
    except Exception as e:
        # その他の予期しないエラー
        print(f"[ERROR] Unexpected error processing {path}: {e}")
        raise

def rasterize_svg_to_png(path: Path, width: int = 1024, background: Optional[str] = None) -> Optional[Path]:
    """SVGをPNGへラスタライズ。cairosvg または inkscape を利用。
    background: '#ffffff' などのCSSカラー。Noneで透過。
    """
    try:
        # 1) cairosvg があれば最優先
        try:
            import cairosvg  # type: ignore
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            kwargs = { 'url': str(path), 'write_to': tmp.name, 'output_width': width }
            if background:
                kwargs['background_color'] = background
            cairosvg.svg2png(**kwargs)
            return Path(tmp.name)
        except Exception:
            pass
        # 2) inkscape CLI
        inkscape = shutil.which('inkscape')
        if inkscape:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            cmd = [inkscape, str(path), '--export-type=png', f'--export-width={width}', f'--export-filename={tmp.name}']
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if background:
                # 透過PNGに背景合成（Pillow）
                try:
                    from PIL import Image as _PILImage
                    img = _PILImage.open(tmp.name).convert('RGBA')
                    bg = _PILImage.new('RGBA', img.size, background)
                    bg.alpha_composite(img)
                    bg = bg.convert('RGB')
                    bg.save(tmp.name, 'PNG')
                except Exception:
                    pass
            return Path(tmp.name)
    except Exception as e:
        print(f"[WARNING] SVG rasterize failed: {e}")
    return None

def encode_image_to_base64(path: Path) -> str:
    """画像ファイルをBase64エンコード"""
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def parse_response_text(response_json: dict) -> dict:
    """
    LM StudioのAPIレスポンスからテキストを取り出し、JSONとして解析。
    フォーマット:
    {
        "caption": "画像の説明",
        "tags": ["タグ1", "タグ2", ...]
    }
    解析に失敗した場合はデフォルト値を返す。
    """
    try:
        content = response_json['choices'][0]['message']['content']
        
        # contentの前後空白を削除
        content = content.strip()
        
        # コードブロックの除去
        if content.startswith('```json'):
            content = content[7:]
        elif content.startswith('```'):
            content = content[3:]
        
        if content.endswith('```'):
            content = content[:-3]
        
        # 再度前後空白を削除
        content = content.strip()
        
        # JSONデコード
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"[WARNING] JSON decode error: {e}")
            print(f"[WARNING] Original content: {content}")
            # よくあるパターン：シングルクォートの置換
            content_fixed = content.replace("'", '"')
            try:
                data = json.loads(content_fixed)
            except:
                raise e
        
        # デフォルト値の設定
        caption = data.get('caption', '')
        tags = data.get('tags', [])
        
        # tagsが文字列の場合はリストに変換
        if isinstance(tags, str):
            tags = [tags]
        elif not isinstance(tags, list):
            tags = []
        
        # tagsの各要素が文字列であることを保証
        tags = [str(tag) for tag in tags if tag]
        
        return {
            'caption': caption,
            'tags': tags
        }
        
    except Exception as e:
        print(f"[WARNING] レスポンス解析エラー: {e}")
        print(f"[WARNING] レスポンス全体: {json.dumps(response_json, ensure_ascii=False, indent=2)}")
        return {
            'caption': '',
            'tags': []
        }

# ----------------- LM Studio API 呼び出し -----------------
def call_lmstudio_text(lmstudio_url: str, model: str, prompt: str, timeout: int = 300) -> dict:
    """LM Studioにテキストのみで問い合わせ、JSONを得る。"""
    headers = {"Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
        "temperature": 0.2,
        "max_tokens": 2000
    }
    resp = requests.post(f"{lmstudio_url}/v1/chat/completions", headers=headers, json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()
def call_lmstudio_vlm(
    lmstudio_url: str, model: str, prompt: str, image_path: Path,
    resize_width: int = 1024, quality: int = 85, timeout: int = 300,
    extra_context: str = ""
) -> dict:
    """
    LM Studio OpenAI互換APIでVLMを呼び出し、画像解析結果を取得。
    戻り値: {"raw": APIレスポンス全体, "parsed": {"caption": str, "tags": list}}
    """
    # JPEG変換・リサイズ
    tmp_path = None
    try:
        tmp_path = convert_and_resize_to_jpeg(image_path, width=resize_width, quality=quality)
        base64_image = encode_image_to_base64(tmp_path)
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()

    # API呼び出し本文
    headers = {"Content-Type": "application/json"}
    # 追加コンテキストがあれば先頭に差し込む
    full_prompt = (extra_context + "\n" + prompt) if extra_context else prompt

    # 画像の渡し方: data URL か http URL を選択（LMSTUDIO_IMAGE_MODE=http で強制）
    img_mode = os.getenv("LMSTUDIO_IMAGE_MODE", "data").lower().strip()
    image_ref = None
    if img_mode == "http":
        try:
            scan_root = os.getenv("SCAN_PATH", "")
            server_host = os.getenv("SERVER_HOST", "127.0.0.1")
            server_port = os.getenv("PORT", "3333")
            rel = None
            if scan_root:
                try:
                    rel = str(Path(image_path).resolve().relative_to(Path(scan_root).resolve()))
                except Exception:
                    rel = Path(image_path).name
            else:
                rel = Path(image_path).name
            rel_web = rel.replace(os.sep, "/")
            image_ref = f"http://{server_host}:{server_port}/{rel_web}"
        except Exception:
            image_ref = f"data:image/jpeg;base64,{base64_image}"
    else:
        image_ref = f"data:image/jpeg;base64,{base64_image}"
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_ref}
                    }
                ]
            }
        ],
        "temperature": 0.25,
        "max_tokens": 4000
    }

    start_time = time.time()
    safe_debug_print(f"API呼び出し: {image_path}")

    try:
        # まずは Chat Completions に送る（mac/一部Windows）
        response = requests.post(
            f"{lmstudio_url}/v1/chat/completions",
            headers=headers,
            json=body,
            timeout=timeout
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        print(f"[ERROR] タイムアウト ({timeout}秒) - {image_path}")
        raise
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] API呼び出しエラー (chat/completions): {e}")
        try:
            if hasattr(e, 'response') and e.response is not None:
                print(f"[ERROR] 応答ボディ: {e.response.text}")
        except Exception:
            pass
        raise
    
    elapsed = time.time() - start_time
    safe_debug_print(f"API応答時間: {elapsed:.2f}秒")

    # レスポンス解析
    body = response.json()
    
    return {"raw": body, "parsed": parse_response_text(body)}

def call_lmstudio_vlm_multi(
    lmstudio_url: str, model: str, prompt: str, image_paths: List[Path],
    timeout: int = 300
) -> dict:
    """複数画像（データURL）を1リクエストで渡して解析。"""
    headers = {"Content-Type": "application/json"}
    contents = [{"type": "text", "text": prompt}]
    for ip in image_paths:
        try:
            mime = 'image/png'
            with open(ip, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
            contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"}
            })
        except Exception:
            continue
    body = {
        "model": model,
        "messages": [{"role": "user", "content": contents}],
        "temperature": 0.25,
        "max_tokens": 3000
    }
    try:
        resp = requests.post(f"{lmstudio_url}/v1/chat/completions", headers=headers, json=body, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] API呼び出しエラー(multi): {e}")
        raise
    raw = resp.json()
    return {"raw": raw, "parsed": parse_response_text(raw)}

# ----------------- Meilisearch操作 -----------------
def get_next_media_id(meili_url: str, meili_key: str, meili_index: str) -> int:
    """Meilisearchから次のmedia_idを取得"""
    try:
        # 最大のIDを持つドキュメントを検索
        url = f"{meili_url}/indexes/{meili_index}/search"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {meili_key}'
        }
        body = {
            'q': '',
            'limit': 1,
            'sort': ['id:desc']
        }
        
        safe_debug_print(f"Getting next ID from: {url}")
        response = requests.post(url, json=body, headers=headers)
        safe_debug_print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            safe_debug_print("Response", result)
            if result['hits']:
                next_id = result['hits'][0]['id'] + 1
                safe_debug_print(f"Max ID found: {result['hits'][0]['id']}, returning: {next_id}")
                return next_id
        else:
            print(f"[ERROR] Meilisearch error response: {response.text}")
        
        # ドキュメントがない場合は1から開始
        safe_debug_print("No documents found, returning ID: 1")
        return 1
    except Exception as e:
        print(f"[ERROR] get_next_media_id failed: {e}")
        return 1

def upsert_meilisearch(meili_url: str, meili_key: str, meili_index: str, 
                      media_id: int, path: str, path_hash: str, caption: str, 
                      tags: List[str], model: str, raw_json: str):
    """MeilisearchにドキュメントをアップサートPSPORT"""
    try:
        # ドキュメントを作成
        import os as _os
        _name = _os.path.basename(path)
        _stem = _name.rsplit('.', 1)[0]
        doc = {
            'id': media_id,
            'path': path,
            'name': _name,
            'path_hash': path_hash,
            'caption': caption,
            'tags': tags,
            'model': model,
            'raw_json': raw_json,
            'searchable_text': f"{_stem} {caption} {' '.join(tags)}"  # 検索用結合テキスト
        }
        
        safe_debug_print("Meilisearch document", doc)
        
        # Meilisearch APIにPOST
        url = f"{meili_url}/indexes/{meili_index}/documents"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {meili_key}'
        }
        
        safe_debug_print(f"Meilisearch request URL: {url}")
        safe_debug_print("Meilisearch headers", headers)
        
        response = requests.post(url, json=[doc], headers=headers)
        
        safe_debug_print(f"Meilisearch response status: {response.status_code}")
        safe_debug_print(f"Meilisearch response text: {response.text}")
        
        if response.status_code == 202:  # Meilisearchは202 Acceptedを返す
            safe_debug_print(f"Meilisearch indexed: {path}")
        else:
            print(f"[ERROR] Meilisearch error: {response.status_code} {response.text}")
            
    except Exception as e:
        print(f"[ERROR] Meilisearch indexing failed: {e}")
        # エラーは発生するが処理は続行する

def get_existing_media_by_path_hash(meili_url: str, meili_key: str, meili_index: str, 
                                   path_hash: str) -> Optional[dict]:
    """path_hashで既存のメディアを検索"""
    try:
        url = f"{meili_url}/indexes/{meili_index}/search"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {meili_key}'
        }
        body = {
            'q': '',
            'filter': f'path_hash = "{path_hash}"',
            'limit': 1
        }
        
        safe_debug_print(f"Searching existing media with path_hash: {path_hash}")
        safe_debug_print(f"Filter: {body['filter']}")
        
        response = requests.post(url, json=body, headers=headers)
        safe_debug_print(f"Search response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            safe_debug_print("Search result", result)
            if result['hits']:
                safe_debug_print("Found existing media", result['hits'][0])
                return result['hits'][0]
            else:
                safe_debug_print("No existing media found")
        else:
            print(f"[ERROR] Search error: {response.text}")
        return None
    except Exception as e:
        print(f"[ERROR] get_existing_media_by_path_hash failed: {e}")
        return None

# ----------------- Qdrant操作 -----------------
def ensure_qdrant_collection(qdrant_client: QdrantClient, collection_name: str, vector_size: int):
    """Qdrantコレクションが存在しない場合は作成"""
    try:
        # コレクションの存在確認
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if collection_name not in collection_names:
            safe_debug_print(f"Creating Qdrant collection: {collection_name}")
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            safe_debug_print(f"Qdrant collection created: {collection_name}")
        
    except Exception as e:
        print(f"[WARNING] Qdrant collection setup failed: {e}")

def upsert_qdrant_vector(qdrant_client: QdrantClient, collection_name: str, 
                        media_id: int, path: str, caption: str, tags: List[str], 
                        model: str, embedding_vector: List[float]):
    """Qdrantにベクトルをアップサート"""
    try:
        # メタデータを作成
        payload = {
            'media_id': media_id,
            'path': path,
            'caption': caption,
            'tags': tags,
            'model': model,
            'text': f"{caption} {' '.join(tags)}"  # 検索用テキスト
        }
        
        safe_debug_print(f"Creating point with ID: {media_id}, vector dim: {len(embedding_vector)}")
        
        # ベクトルをアップサート
        point = PointStruct(
            id=media_id,  # 既に整数IDとして渡される
            vector=embedding_vector,
            payload=payload
        )
        
        result = qdrant_client.upsert(
            collection_name=collection_name,
            points=[point],
            wait=True
        )
        
        safe_debug_print(f"Qdrant upsert result: {result}")
        safe_debug_print(f"Qdrant vector upserted: {path}")
        
        # 確認のためにポイントを取得してみる
        try:
            retrieved_point = qdrant_client.retrieve(
                collection_name=collection_name,
                ids=[media_id]
            )
            safe_debug_print(f"Retrieved point: {len(retrieved_point)} points")
        except Exception as retrieve_error:
            print(f"[WARNING] Could not retrieve point: {retrieve_error}")
        
    except Exception as e:
        print(f"[ERROR] Qdrant vector upsert failed: {e}")
        import traceback
        traceback.print_exc()

# ----------------- 処理メイン -----------------
def process_image(
    sb_model, qdrant_client,
    meili_url: str, meili_key: str, meili_index: str,
    collection_name: str, p: Path, rel_base: Optional[Path], args
):
    # path 保存: override_path があればそれを優先。なければ rel_base 相対 or 絶対
    override_rel = getattr(args, 'override_path', None)
    if isinstance(override_rel, str) and override_rel.strip():
        rel = override_rel.strip()
    else:
        if rel_base:
            try:
                rel = str(p.relative_to(rel_base))
            except Exception:
                rel = str(p.resolve())
        else:
            rel = str(p.resolve())

    # path_hashを計算してIDを生成
    path_hash = path_sha256_hex(rel)
    # IDスキームに従ってIDを生成（既定: rel8）
    media_id = compute_media_id_from_rel(rel)
    safe_debug_print(f"Processing: {rel}")
    safe_debug_print(f"Path hash: {path_hash}")
    safe_debug_print(f"Using converted ID (64-bit): {media_id}")

    try:
        # SVGファイルはロゴ/マーク解析モード（まずは二背景ラスタライズ→VLM、失敗時テキスト解析）
        if str(p).lower().endswith('.svg'):
            try:
                svg_text = p.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                svg_text = ''
            # 単純抽出: <text>要素、<title>、<desc>、fillカラー
            import re
            texts = re.findall(r'<text[^>]*>(.*?)</text>', svg_text, flags=re.I|re.S)
            titles = re.findall(r'<title[^>]*>(.*?)</title>', svg_text, flags=re.I|re.S)
            descs = re.findall(r'<desc[^>]*>(.*?)</desc>', svg_text, flags=re.I|re.S)
            colors = re.findall(r'fill=["\'](#?[0-9a-fA-F]{3,8})["\']', svg_text)
            ids = re.findall(r'id=["\']([^"\']+)["\']', svg_text)
            classes = re.findall(r'class=["\']([^"\']+)["\']', svg_text)
            has_anim = bool(re.search(r'<animate|<animatetransform|animate\-', svg_text, flags=re.I))
            # 正規化
            def _clean(s):
                return re.sub(r'\s+', ' ', s).strip()
            texts = [_clean(t) for t in texts if _clean(t)]
            titles = [_clean(t) for t in titles if _clean(t)]
            descs = [_clean(t) for t in descs if _clean(t)]
            uniq_colors = []
            for c in colors:
                c = c.lower()
                if c not in uniq_colors:
                    uniq_colors.append(c)
            id_tokens = []
            for s in (ids + classes):
                for tok in re.split(r'[\s_\-]+', s):
                    tok = tok.strip()
                    if tok and tok not in id_tokens and len(tok) <= 32:
                        id_tokens.append(tok)
            file_stem = Path(rel).name.rsplit('.', 1)[0]
            # プロンプト
            svg_snippet = svg_text[:2000]
            context = """
あなたはロゴ解析の専門家です。以下のSVGロゴを解析し、日本語でJSONのみ出力してください。
厳守事項:
- これは静止画のロゴです。『ローディング/読み込み/スピナー/進捗』等の語は使わない（特に円形でも）。
- 文字が含まれている場合はロゴの文字列をできるだけ正確に読み取り、captionに含める。
- 形状（円/角/記号）よりも、文字情報とブランド名/略称を優先して記述。
- 色（代表色）も補足。抽出ヒントやファイル名から推測できるブランド名があれば活用。
- 出力は厳密にJSONのみ。余分な文章は出力しない。

出力形式:
{
  "caption": "ロゴの内容（文字列・ブランド名・色・象徴を簡潔に。例: ‘白地に黒の“ACME”ロゴ’）",
  "tags": ["ロゴ", "文字(ACMEなど)", "色(黒/白/#000000など)", "スタイル/形状"]
}
""".strip()
            hint = {
                "extracted_text": texts[:5],
                "title": titles[:3],
                "desc": descs[:3],
                "colors": uniq_colors[:6],
                "ids": id_tokens[:6],
                "file_stem": file_stem,
                "has_animation": has_anim
            }
            prompt = f"{context}\n\n抽出ヒント: {json.dumps(hint, ensure_ascii=False)}\n\nSVGの一部（テキスト化）:\n{svg_snippet}"
            # 先に白/黒背景でラスタライズしてVLMに送る
            png_white = rasterize_svg_to_png(p, width=args.resize_width, background='#ffffff')
            png_black = rasterize_svg_to_png(p, width=args.resize_width, background='#000000')
            if png_white and png_white.exists() and png_black and png_black.exists():
                try:
                    # ローディング誤判定を強く抑制する追加制約
                    ctx_rules = (
                        "これは静止画のロゴです。『ローディング/読み込み/スピナー/進捗』という表現は禁止。"
                        "2枚の画像は可視化のため白背景と黒背景でレンダリングした同一ロゴです。"
                        "背景は本来のロゴ背景ではないため、前景（ロゴ）の色・文字を優先して記述してください。"
                        "両背景で共通して見える形状と文字列をロゴ本体として解釈してください。"
                    )
                    logo_prompt = prompt + "\n\n追加制約:\n" + ctx_rules
                    res = call_lmstudio_vlm_multi(
                        args.lmstudio_url, args.model, logo_prompt, [png_white, png_black], timeout=args.timeout
                    )
                    parsed = res.get('parsed', {"caption": "", "tags": []})
                    raw = json.dumps(res.get('raw', {}), ensure_ascii=False)
                    caption = parsed.get('caption') or 'SVGロゴ/マーク'
                    tags = parsed.get('tags') or []
                    if 'ロゴ' not in tags:
                        tags.append('ロゴ')
                    # 文字列ヒントもタグに補完
                    if texts:
                        for t in texts[:2]:
                            t = _clean(t)
                            if t and t not in tags:
                                tags.append(t)
                    if file_stem and file_stem not in tags:
                        tags.append(file_stem)
                except Exception as e:
                    safe_debug_print(f"[INFO] SVG rasterized VLM failed, fallback to text: {e}")
                finally:
                    try:
                        png_white.unlink()
                    except Exception:
                        pass
                    try:
                        png_black.unlink()
                    except Exception:
                        pass
            if not locals().get('caption'):
                # テキストのみでLMに投げるフォールバック
                try:
                    raw_json = call_lmstudio_text(args.lmstudio_url, args.model, prompt, timeout=max(120, args.timeout))
                    parsed = parse_response_text(raw_json)
                    caption = parsed.get('caption') or 'SVGロゴ/マーク'
                    tags = parsed.get('tags') or []
                    if 'ロゴ' not in tags:
                        tags.append('ロゴ')
                    raw = json.dumps(raw_json, ensure_ascii=False)
                except Exception as e:
                    safe_debug_print(f"[INFO] SVG text analysis failed, using heuristic: {e}")
                    base = ' / '.join(titles[:1] + texts[:1]) or 'SVGロゴ/マーク'
                    caption = f"{base}（SVGベクター）"
                    tags = ['ロゴ', 'svg', 'ベクター'] + (titles[:1] or [])
                    raw = json.dumps({"note": "SVG heuristic", "hint": hint}, ensure_ascii=False)
        else:
            # 3Dスナップショット等の文脈があればプロンプトに付与
            ctx = getattr(args, 'context', '') or ''
            res = call_lmstudio_vlm(
                args.lmstudio_url, args.model, PROMPT, p,
                resize_width=args.resize_width, quality=args.quality, timeout=args.timeout,
                extra_context=ctx
            )
            parsed = res.get("parsed", {"caption": "", "tags": []})
            raw = json.dumps(res.get("raw", {}), ensure_ascii=False)  # JSON丸ごと保存

            caption = parsed.get("caption", "")
            tags = parsed.get("tags", [])
        # タグの整形
        if not isinstance(tags, list):
            tags = [str(tags)] if tags else []
        tags = [str(t) for t in tags if t is not None]
        # 追加タグを強制付与
        extra = getattr(args, 'extra_tag', None) or []
        for t in extra:
            if t is None:
                continue
            constag = str(t)
            if constag not in tags:
                tags.append(constag)

        # Meilisearch indexing
        safe_debug_print(f"Starting Meilisearch indexing for {rel}")
        try:
            upsert_meilisearch(meili_url, meili_key, meili_index, media_id, rel, path_hash, 
                              caption, tags, args.model, raw)
            safe_debug_print(f"Meilisearch indexing completed for {rel}")
        except Exception as e:
            print(f"[WARNING] Meilisearch indexing failed for {rel}: {e}")
            # トレースバックを表示するが処理は続行
        
        # Qdrant vector embedding and storage
        safe_debug_print(f"Starting Qdrant embedding for {rel}")
        try:
            # テキストをエンベディング（ファイル名ベースも含める）
            import os as _os
            _name = _os.path.basename(rel)
            _stem = _os.path.splitext(_name)[0]
            search_text = f"{_stem} {caption} {' '.join(tags)}"
            safe_debug_print(f"Embedding text: {search_text}")
            
            if "plamo-embedding" in args.emb_model.lower():
                # Plamo embeddingの場合
                import torch
                with torch.inference_mode():
                    # encode_documentメソッドを使用
                    embedding_tensor = plamo_model.encode_document([search_text], plamo_tokenizer)
                    embedding = embedding_tensor[0].cpu().numpy()
                    safe_debug_print(f"Plamo embedding generated, size: {len(embedding)}")
            else:
                # 通常のsentence-transformers
                embedding = sb_model.encode([search_text], convert_to_numpy=True, normalize_embeddings=True)[0]
                safe_debug_print(f"Sentence-transformers embedding generated, size: {len(embedding)}")
            
            # Qdrantにベクトル保存
            safe_debug_print(f"Upserting to Qdrant with media_id: {media_id}")
            upsert_qdrant_vector(qdrant_client, collection_name, media_id, rel, caption, tags, args.model, embedding.tolist())
            safe_debug_print(f"Qdrant embedding completed for {rel}")
        except Exception as e:
            print(f"[WARNING] Qdrant embedding failed for {rel}: {e}")
            # トレースバックを表示するが処理は続行

        print(f"[OK] {p} -> {tags}")
        
        # 解析結果をJSONで出力（server.jsで使用）
        result_json = {
            "success": True,
            "path": rel,
            "caption": caption,
            "tags": tags,
            "model": args.model,
            "media_id": media_id
        }
        print(f"RESULT_JSON:{json.dumps(result_json, ensure_ascii=False)}")

    except Exception as e:
        print(f"[ERROR] {p} 処理失敗: {e}")
        error_json = {
            "success": False,
            "path": rel if 'rel' in locals() else str(p),
            "error": str(e)
        }
        print(f"RESULT_JSON:{json.dumps(error_json, ensure_ascii=False)}")
        import traceback
        traceback.print_exc()

def main():
    global LMSTUDIO_URL, LMSTUDIO_MODEL
    global MEILI_URL, MEILI_API_KEY, MEILI_INDEX, QDRANT_URL, QDRANT_COLLECTION, EMB_MODEL

    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, help="単一画像ファイル")
    p.add_argument("--media_dir", type=Path, help="画像/動画ディレクトリ（再帰）")
    p.add_argument("--rel_base", type=Path, help="相対パス基準ディレクトリ")
    # 実体ファイルとは別に、登録時の論理パスを上書きしたい場合に使用（例: GLBに対するPNGスナップショット解析）
    p.add_argument("--override_path", type=str, help="登録時の論理パスを上書き（rel_baseからの相対推奨）")
    
    p.add_argument("--lmstudio_url", type=str, default=LMSTUDIO_URL)
    p.add_argument("--model", type=str, default=LMSTUDIO_MODEL)
    p.add_argument("--resize_width", type=int, default=DEFAULT_RESIZE)
    p.add_argument("--quality", type=int, default=DEFAULT_QUALITY)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    
    p.add_argument("--meili_url", type=str, default=MEILI_URL)
    p.add_argument("--meili_key", type=str, default=MEILI_API_KEY)
    p.add_argument("--meili_index", type=str, default=MEILI_INDEX)
    
    p.add_argument("--qdrant_url", type=str, default=QDRANT_URL)
    p.add_argument("--qdrant_collection", type=str, default=QDRANT_COLLECTION)
    
    p.add_argument("--emb_model", type=str, default=EMB_MODEL)
    # 解析結果に強制付与するタグ（複数指定可）
    p.add_argument("--extra_tag", action='append', default=None, help="解析結果に強制付与するタグ（複数指定可）")
    # プロンプトに追加する文脈（3Dスナップショット等の説明）
    p.add_argument("--context", type=str, default="", help="プロンプトに付与する追加コンテキスト")

    args = p.parse_args()

    # グローバル変数に反映
    LMSTUDIO_URL = args.lmstudio_url
    LMSTUDIO_MODEL = args.model

    # sentence-transformers (Plamo以外の場合のみ)
    if "plamo-embedding" not in args.emb_model.lower():
        sb_model = SentenceTransformer(args.emb_model)
    else:
        sb_model = None  # Plamoの場合は後で初期化
    
    # Qdrant
    qdrant_client = QdrantClient(url=args.qdrant_url)
    
    # モデルに応じたエンコード方法を選択
    if "plamo-embedding" in args.emb_model.lower():
        # Plamo embeddingの場合は特別な処理が必要
        import torch
        from transformers import AutoTokenizer, AutoModel
        
        # 信頼できるモデルのホワイトリスト
        trusted_models = [
            "pfnet/plamo-embedding-1b",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2"
        ]
        
        trust_code = args.emb_model in trusted_models
        if not trust_code:
            print(f"[SECURITY WARNING] Untrusted model: {args.emb_model}. Using with trust_remote_code=False")
        
        tokenizer = AutoTokenizer.from_pretrained(args.emb_model, trust_remote_code=trust_code)
        model = AutoModel.from_pretrained(args.emb_model, trust_remote_code=trust_code)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        # ダミーエンコード（次元数確認用）
        with torch.inference_mode():
            dummy_embedding = model.encode_document(["dummy"], tokenizer)
            dummy = dummy_embedding[0].cpu().numpy()
        
        vector_size = len(dummy)
        safe_debug_print(f"Using Plamo embedding model with {vector_size} dimensions on {device}")
        
        # 後で使うためにグローバルに保存
        global plamo_tokenizer, plamo_model
        plamo_tokenizer = tokenizer
        plamo_model = model
    else:
        # 通常のsentence-transformers
        dummy = sb_model.encode(["dummy"], convert_to_numpy=True, normalize_embeddings=True)[0]
        vector_size = len(dummy)
    
    # Qdrantコレクションを確保
    ensure_qdrant_collection(qdrant_client, args.qdrant_collection, vector_size)

    if args.image:
        pth = args.image.resolve()
        rel_base = args.rel_base.resolve() if args.rel_base else None
        process_image(sb_model, qdrant_client, MEILI_URL, MEILI_API_KEY, MEILI_INDEX, QDRANT_COLLECTION, pth, rel_base, args)

    elif args.media_dir:
        # ディレクトリ処理
        exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        files = []
        for ext in exts:
            files.extend(args.media_dir.rglob(f"*{ext}"))
            files.extend(args.media_dir.rglob(f"*{ext.upper()}"))

        rel_base = args.rel_base.resolve() if args.rel_base else args.media_dir.resolve()

        for f in tqdm(files, desc="画像処理"):
            process_image(sb_model, qdrant_client, MEILI_URL, MEILI_API_KEY, MEILI_INDEX, QDRANT_COLLECTION, f, rel_base, args)

    else:
        print("--image または --media_dir を指定してください")
        sys.exit(1)

    print("完了。")

if __name__ == "__main__":
    main()
