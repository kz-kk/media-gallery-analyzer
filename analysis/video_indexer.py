#!/usr/bin/env python3
# coding: utf-8
"""
video_indexer_simple.py
- 既存のmedia_indexer_lmstudio.pyのprocess_image関数をベースに動画解析
- FFmpegでフレーム抽出後、既存システムと同じ処理フロー
"""
from pathlib import Path
import argparse
import json
import sys
import tempfile
import os
import hashlib
import subprocess
import shutil
from typing import Optional, List, Dict
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
import ffmpeg
from tqdm import tqdm

# 既存の画像解析モジュールから必要な関数をインポート
from image_indexer import (
    load_env, convert_and_resize_to_jpeg, call_lmstudio_vlm, 
    parse_response_text, upsert_meilisearch, ensure_qdrant_collection, 
    upsert_qdrant_vector, path_sha256_hex, LMSTUDIO_URL, LMSTUDIO_MODEL, 
    MEILI_URL, MEILI_API_KEY, MEILI_INDEX, QDRANT_URL, QDRANT_COLLECTION, 
    EMB_MODEL, DEFAULT_RESIZE, DEFAULT_QUALITY, DEFAULT_TIMEOUT, PROMPT
)
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from audio_analyzer_v2 import analyze_audio_comprehensive

# 動画対応拡張子
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.mpg', '.mpeg'}

def check_ffmpeg():
    """FFmpegがインストールされているか確認"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  FFmpegがインストールされていません")
        return False

def safe_parse_framerate(framerate_str: str) -> float:
    """フレームレート文字列を安全に解析"""
    if '/' in framerate_str:
        try:
            num, denom = framerate_str.split('/', 1)
            return float(num) / float(denom)
        except (ValueError, ZeroDivisionError):
            return 0.0
    try:
        return float(framerate_str)
    except ValueError:
        return 0.0

def get_video_info(video_path: Path) -> Dict:
    """動画の情報を取得"""
    try:
        probe = ffmpeg.probe(str(video_path))
        video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
        
        if video_stream:
            duration = float(probe['format']['duration'])
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            fps = safe_parse_framerate(video_stream['r_frame_rate'])  # 安全な解析方法
            
            return {
                "duration": duration,
                "width": width,
                "height": height,
                "fps": fps,
                "codec": video_stream.get('codec_name', 'unknown')
            }
    except Exception as e:
        print(f"動画情報取得エラー: {e}")
    
    return {"duration": 0, "width": 0, "height": 0, "fps": 0, "codec": "unknown"}

def has_audio_stream(video_path: Path) -> bool:
    try:
        probe = ffmpeg.probe(str(video_path))
        return any(s for s in probe.get('streams', []) if s.get('codec_type') == 'audio')
    except Exception:
        return False

def extract_audio_track(video_path: Path) -> Optional[Path]:
    """動画から音声トラックを抽出（モノラル16kHz WAV）。なければNone"""
    if not has_audio_stream(video_path):
        return None
    temp_dir = Path(tempfile.mkdtemp())
    out_wav = temp_dir / "audio_track.wav"
    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(str(out_wav), acodec='pcm_s16le', ac=1, ar=16000, vn=None)
            .overwrite_output()
            .run(quiet=True)
        )
        if out_wav.exists() and out_wav.stat().st_size > 0:
            return out_wav
    except Exception as e:
        print(f"音声抽出エラー: {e}")
    # 失敗時はクリーンアップ
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass
    return None

def extract_frames_ffmpeg(video_path: Path, interval_seconds: float = 5.0, max_frames: int = 10) -> List[Path]:
    """FFmpegを使用して動画から定期的にフレームを抽出"""
    temp_dir = Path(tempfile.mkdtemp())
    frames = []
    
    try:
        info = get_video_info(video_path)
        duration = info['duration']
        
        if duration <= 0:
            print(f"⚠️  動画の長さを取得できません: {video_path.name}")
            return []
        
        # 抽出するタイムスタンプを計算
        timestamps = []
        current_time = 1.0  # 1秒から開始
        while current_time < duration and len(timestamps) < max_frames:
            timestamps.append(current_time)
            current_time += interval_seconds
        
        print(f"📹 {len(timestamps)}枚のフレームを抽出中...")
        
        # FFmpegでフレーム抽出
        for i, ts in enumerate(tqdm(timestamps, desc="フレーム抽出")):
            output_path = temp_dir / f"frame_{i:04d}.jpg"
            
            try:
                (
                    ffmpeg
                    .input(str(video_path), ss=ts)
                    .output(str(output_path), vframes=1, format='image2', vcodec='mjpeg')
                    .overwrite_output()
                    .run(quiet=True)
                )
                frames.append(output_path)
            except Exception as e:
                print(f"フレーム抽出エラー (時刻 {ts:.1f}秒): {e}")
                continue
        
        return frames
        
    except Exception as e:
        print(f"フレーム抽出エラー: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return []

def process_video_with_existing_system(video_path: Path, args):
    """既存のprocess_image関数と同じ流れで動画を処理"""
    
    # 動画のパス情報を準備（既存システムと同じ）
    rel = str(video_path.resolve())
    path_hash = path_sha256_hex(rel)
    media_id = int(path_hash[:16], 16)
    
    print(f"[DEBUG] Processing video: {rel}")
    print(f"[DEBUG] Path hash: {path_hash}")
    print(f"[DEBUG] Using converted ID: {media_id}")
    
    try:
        # 1. フレーム抽出
        frames = extract_frames_ffmpeg(video_path, args.interval, args.max_frames)
        if not frames:
            print(f"⚠️  フレーム抽出失敗: {video_path.name}")
            return False
        
        # 2. 各フレームを解析（既存システムと同じ方法）
        all_captions = []
        all_tags = set()
        
        print(f"🔍 {len(frames)}枚のフレームを解析中...")
        for i, frame_path in enumerate(tqdm(frames, desc="フレーム解析")):
            try:
                # 既存のcall_lmstudio_vlm関数を使用
                res = call_lmstudio_vlm(
                    args.lmstudio_url, args.model, PROMPT, frame_path,
                    resize_width=args.resize_width, quality=args.quality, timeout=args.timeout
                )
                
                if res:
                    parsed = res.get("parsed", {"caption": "", "tags": []})
                    if parsed.get("caption"):
                        all_captions.append(f"フレーム{i+1}: {parsed['caption']}")
                    if parsed.get("tags"):
                        all_tags.update(parsed["tags"])
                        
            except Exception as e:
                print(f"フレーム{i+1}の解析エラー: {e}")
                continue
        
        # 3. 動画メタデータを取得
        video_info = get_video_info(video_path)
        duration_str = f"{video_info['duration']:.1f}秒" if video_info['duration'] > 0 else "不明"
        resolution_str = f"{video_info['width']}x{video_info['height']}" if video_info['width'] > 0 else "不明"

        # 3.5 音声解析（存在する場合のみ）
        audio_summary = None
        audio_tags: List[str] = []
        audio_payload: Dict = { "has_audio": False }
        transcript_text_value = ""
        audio_path = extract_audio_track(video_path)
        if audio_path:
            try:
                # Whisperの既定モデルを 'small' に引き上げ（歌唱の日本語認識精度を改善）
                whisper_model = os.getenv('WHISPER_MODEL', 'small')
                audio_result = analyze_audio_comprehensive(audio_path, use_whisper=True, whisper_model=whisper_model)
                # 判定: 無音かどうか簡易チェック
                feats = audio_result.get('features', {})
                rms_mean = float(feats.get('rms_mean') or 0.0)
                transcript_text = ((audio_result.get('transcription') or {}).get('text') or '').strip()
                transcript_text_value = transcript_text
                audio_type = feats.get('audio_type') or ('silent' if (rms_mean < 0.005 and not transcript_text) else 'ambiguous')
                audio_summary = audio_result.get('description')
                audio_tags = list(set(audio_result.get('tags') or []))
                audio_payload = {
                    "has_audio": True,
                    "audio_type": audio_type,
                    "caption": audio_summary,
                    "tags": audio_tags,
                    "has_transcription": bool(transcript_text),
                    "language": (audio_result.get('transcription') or {}).get('language'),
                    "speaker_gender": audio_result.get('speaker_gender')
                }
            except Exception as e:
                print(f"音声解析エラー: {e}")
            finally:
                # 一時WAVをクリーンアップ
                try:
                    if audio_path and audio_path.exists():
                        shutil.rmtree(audio_path.parent, ignore_errors=True)
                except Exception:
                    pass

        # 4. 結果を統合（音声要約もキャプションに反映）
        base_caption = f"動画ファイル ({duration_str})"
        frames_caption = " / ".join(all_captions[:3]) if all_captions else ""
        audio_caption_part = ""
        if audio_payload.get('has_audio'):
            # 歌詞抜粋は常に優先して表示（存在する場合）
            parts = []
            if transcript_text_value:
                snippet = transcript_text_value[:120]
                parts.append(f"歌詞抜粋: 『{snippet}』")
            if audio_summary:
                # 歌詞抜粋がある場合は要約を少し短めに
                limit = 120 if transcript_text_value else 140
                short_audio = audio_summary if len(audio_summary) <= limit else audio_summary[:limit] + '…'
                # 話者性別表記
                gender = audio_payload.get('speaker_gender') or ''
                gender_text = "（話者: 男性）" if gender == 'male' else ("（話者: 女性）" if gender == 'female' else '')
                parts.append(f"音声要約: {short_audio}{gender_text}")
            audio_caption_part = " / ".join(parts)
        parts = [p for p in [base_caption, frames_caption, audio_caption_part] if p]
        combined_caption = " : ".join(parts)

        combined_tags = list(all_tags) + ["動画", "video", duration_str, resolution_str, video_info['codec']]
        if audio_payload.get('has_audio'):
            atype = audio_payload.get('audio_type')
            if atype == 'speech':
                combined_tags += ["音声あり", "会話"]
            elif atype == 'music':
                combined_tags += ["音声あり", "楽曲", "music"]
                if audio_payload.get('has_transcription'):
                    combined_tags += ["歌詞あり", "ボーカル"]
            elif atype == 'sfx':
                combined_tags += ["音声あり", "効果音"]
            elif atype == 'silent':
                combined_tags += ["無音"]
            # 話者性別タグ
            gender = audio_payload.get('speaker_gender')
            if gender == 'male':
                combined_tags += ["男性声"]
            elif gender == 'female':
                combined_tags += ["女性声"]
            # 解析タグも一部足す（多すぎないように）
            combined_tags += audio_tags[:5]

        # 5. 結果をJSONとして作成（既存システム互換）
        result_json = {
            "caption": combined_caption,
            "tags": combined_tags,
            "video_metadata": video_info,
            "analyzed_frames": len(frames),
            "audio": audio_payload
        }
        raw_json = json.dumps(result_json, ensure_ascii=False)
        
        # 6. Meilisearchに保存（既存システムと同じ）
        print(f"[DEBUG] Starting Meilisearch indexing for {rel}")
        try:
            upsert_meilisearch(
                MEILI_URL, MEILI_API_KEY, MEILI_INDEX, 
                media_id, rel, path_hash, 
                combined_caption, combined_tags, args.model, raw_json
            )
            print(f"[DEBUG] Meilisearch indexing completed for {rel}")
        except Exception as e:
            print(f"[WARNING] Meilisearch indexing failed for {rel}: {e}")
        
        # 7. Qdrantに保存（既存システムと完全に同じロジック）
        print(f"[DEBUG] Starting Qdrant embedding for {rel}")
        try:
            qdrant_client = QdrantClient(url=QDRANT_URL)
            
            # 環境変数からモデル名を取得（既存システムと同じ）
            emb_model_name = os.getenv("EMB_MODEL", "all-MiniLM-L6-v2")
            
            # テキストをエンベディング（音声説明/タグも含める）
            audio_text = (audio_summary or '') + ' ' + ' '.join(audio_tags)
            search_text = f"{combined_caption} {' '.join(combined_tags)} {audio_text}"
            print(f"[DEBUG] Embedding text: {search_text}")
            
            if "plamo-embedding" in emb_model_name.lower():
                # Plamo embeddingの場合（既存システムと同じ処理）
                import torch
                from transformers import AutoTokenizer, AutoModel
                
                # 信頼できるモデルのホワイトリスト
                trusted_models = [
                    "pfnet/plamo-embedding-1b",
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers/all-mpnet-base-v2"
                ]
                
                trust_code = emb_model_name in trusted_models
                if not trust_code:
                    print(f"[SECURITY WARNING] Untrusted model: {emb_model_name}. Using trust_remote_code=False")
                
                tokenizer = AutoTokenizer.from_pretrained(emb_model_name, trust_remote_code=trust_code)
                plamo_model = AutoModel.from_pretrained(emb_model_name, trust_remote_code=trust_code)
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                plamo_model = plamo_model.to(device)
                
                with torch.inference_mode():
                    # ドキュメントエンベディングにはencode_documentメソッドを使用
                    embedding_tensor = plamo_model.encode_document([search_text], tokenizer)
                    embedding = embedding_tensor[0].cpu().numpy()
                    print(f"[DEBUG] Plamo embedding generated, size: {len(embedding)}")
                    
                # コレクションを確保（Plamo embeddingのサイズに合わせて）
                ensure_qdrant_collection(qdrant_client, args.qdrant_collection, len(embedding))
            else:
                # 通常のsentence-transformers（既存システムと同じ処理）
                sb_model = SentenceTransformer(emb_model_name)
                # Cosine安定化のため正規化
                emb = sb_model.encode([search_text], convert_to_numpy=True, normalize_embeddings=True)[0]
                embedding = emb / (np.linalg.norm(emb) or 1.0)
                print(f"[DEBUG] Sentence-transformers embedding generated, size: {len(embedding)}")
                
                # コレクションを確保
                ensure_qdrant_collection(qdrant_client, args.qdrant_collection, len(embedding))
            
            # Qdrantにベクトル保存
            upsert_qdrant_vector(
                qdrant_client, args.qdrant_collection, media_id, rel, 
                combined_caption, combined_tags, args.model, embedding.tolist()
            )
            print(f"[DEBUG] Qdrant embedding completed for {rel}")
            
        except Exception as e:
            print(f"[WARNING] Qdrant embedding failed for {rel}: {e}")
        
        # 成功メッセージ
        print(f"✅ 登録完了: {video_path.name}")
        print(f"   キャプション: {combined_caption[:100]}...")
        print(f"   タグ: {', '.join(combined_tags[:10])}")
        
        # server.js用の構造化されたJSON結果を出力
        video_result = {
            "success": True,
            "file_name": video_path.name,
            "caption": combined_caption,
            "tags": combined_tags,
            "video_metadata": video_info,
            "analyzed_frames": len(frames),
            "media_id": media_id,
            "model": args.model
        }
        print(f"VIDEO_ANALYSIS_RESULT: {json.dumps(video_result, ensure_ascii=False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ エラー: {video_path.name}")
        print(f"   詳細: {str(e)}")
        
        # エラー時もJSONで出力
        error_result = {
            "success": False,
            "file_name": video_path.name,
            "error": str(e)
        }
        print(f"VIDEO_ANALYSIS_RESULT: {json.dumps(error_result, ensure_ascii=False)}")
        
        return False
        
    finally:
        # 一時フレームファイルをクリーンアップ
        if 'frames' in locals() and frames:
            temp_dir = frames[0].parent
            shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description="動画ファイルから解析（既存システム準拠）")
    parser.add_argument("--video", type=str, help="単一動画ファイルのパス")
    parser.add_argument("--video_dir", type=str, help="動画ディレクトリのパス")
    parser.add_argument("--lmstudio_url", type=str, default=LMSTUDIO_URL)
    parser.add_argument("--model", type=str, default=LMSTUDIO_MODEL)
    parser.add_argument("--resize_width", type=int, default=DEFAULT_RESIZE)
    parser.add_argument("--quality", type=int, default=DEFAULT_QUALITY)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--interval", type=float, default=5.0, help="フレーム抽出間隔（秒）")
    parser.add_argument("--max_frames", type=int, default=10, help="最大抽出フレーム数")
    parser.add_argument("--emb_model", type=str, default=EMB_MODEL)
    parser.add_argument("--qdrant_collection", type=str, default=QDRANT_COLLECTION)
    
    args = parser.parse_args()
    
    if not args.video and not args.video_dir:
        parser.print_help()
        sys.exit(1)
    
    # FFmpegチェック
    if not check_ffmpeg():
        sys.exit(1)
    
    # 環境変数読み込み
    load_env()
    
    # 処理対象の動画ファイルリスト
    video_files = []
    
    if args.video:
        video_path = Path(args.video)
        if video_path.exists() and video_path.suffix.lower() in VIDEO_EXTENSIONS:
            video_files.append(video_path)
        else:
            print(f"⚠️  動画ファイルが見つかりません: {args.video}")
    
    if args.video_dir:
        video_dir = Path(args.video_dir)
        if video_dir.exists():
            for ext in VIDEO_EXTENSIONS:
                video_files.extend(video_dir.rglob(f"*{ext}"))
                video_files.extend(video_dir.rglob(f"*{ext.upper()}"))
        else:
            print(f"⚠️  ディレクトリが見つかりません: {args.video_dir}")
    
    video_files = list(set(video_files))
    print(f"📊 {len(video_files)}個の動画ファイルを処理します")
    
    success_count = 0
    # 各動画を処理
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}]")
        if process_video_with_existing_system(video_path, args):
            success_count += 1
    
    print(f"\n✅ 処理完了: {success_count}/{len(video_files)}個の動画を正常に処理しました")

if __name__ == "__main__":
    main()
