#!/usr/bin/env python3
"""
必要なモデルをダウンロードするセットアップスクリプト

既定で行うこと:
- Whisper "small" の事前取得
- （任意）Plamo 1B の事前取得（巨大・時間がかかる）
- （任意）Sentence-Transformers の埋め込みモデル事前取得
- （任意）PANNs(AudioSet) の重みをウォームアップ（panns_inference がある場合）

使い方例:
  python setup_models.py                # Whisperのみ
  ENABLE_PLAMO=1 python setup_models.py # Plamoも取得
  EMB_MODEL=sentence-transformers/all-MiniLM-L6-v2 python setup_models.py  # STモデル取得
  ENABLE_PANNS=1 python setup_models.py # PANNsをウォームアップ（要ネットワーク）
"""
import os
import sys
from pathlib import Path

def download_models():
    print("=" * 60)
    print("Media Gallery Analyzer - モデルセットアップ")
    print("=" * 60)
    
    # Plamoモデルのダウンロード（Hugging Faceキャッシュを使用、既定で実施）
    print("\n1. Plamo埋め込みモデルの確認...")
    try:
        from transformers import AutoTokenizer, AutoModel
        model_name = "pfnet/plamo-embedding-1b"
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        cached = False
        for item in hf_cache.glob("models--pfnet--plamo-embedding-1b*"):
            if item.is_dir():
                cached = True
                print(f"   ✓ 既にキャッシュ済み: {model_name}")
                break
        if not cached:
            print(f"   ダウンロード中: {model_name}")
            print("   初回ダウンロードには時間がかかります（約1.8GB）...")
            AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            AutoModel.from_pretrained(model_name, trust_remote_code=True)
            print(f"   ✓ ダウンロード完了")
        print(f"   保存先: {hf_cache}")
    except Exception as e:
        print(f"   ✗ エラー: {e}")
        print("   注: 埋め込みモデルのダウンロードに失敗しました。")
        print("   音声解析は動作しますが、検索精度が低下する可能性があります。")
    
    # Whisperモデルの確認
    print("\n2. Whisperモデルの確認...")
    try:
        import whisper
        
        # デフォルトで使用するモデル
        default_model = "small"
        whisper_cache = Path.home() / ".cache" / "whisper"
        
        # smallモデルを自動ダウンロード
        print(f"   デフォルトモデル（{default_model}）を確認中...")
        model_file = whisper_cache / f"{default_model}.pt"
        
        if model_file.exists():
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"   ✓ {default_model}モデル: {size_mb:.1f} MB（キャッシュ済み）")
        else:
            print(f"   {default_model}モデルをダウンロード中...")
            try:
                model = whisper.load_model(default_model)
                print(f"   ✓ {default_model}モデルのダウンロード完了")
            except Exception as e:
                print(f"   ✗ {default_model}モデルのダウンロードに失敗: {e}")
        
        # インストール済みモデルの確認
        print("\n   インストール済みのWhisperモデル:")
        whisper_models = ["tiny", "base", "small", "medium", "large"]
        
        for model_size in whisper_models:
            model_file = whisper_cache / f"{model_size}.pt"
            if model_file.exists():
                size_mb = model_file.stat().st_size / (1024 * 1024)
                if model_size == default_model:
                    print(f"   ✓ {model_size}: {size_mb:.1f} MB [デフォルト]")
                else:
                    print(f"   ✓ {model_size}: {size_mb:.1f} MB")
            else:
                print(f"   - {model_size}: 未インストール")
        
        print(f"\n   保存先: {whisper_cache}")
        print("   注: 'small'モデル（約240MB）- 速度と精度のバランスが最適")
    except ImportError:
        print("   ✗ Whisperがインストールされていません")
    
    # Sentence-Transformers の埋め込みモデル取得（既定で実施）
    print("\n3. Sentence-Transformers埋め込みモデルの確認...")
    try:
        emb_name = os.getenv('EMB_MODEL') or 'sentence-transformers/all-MiniLM-L6-v2'
        from sentence_transformers import SentenceTransformer
        print(f"   ロード中: {emb_name}")
        st = SentenceTransformer(emb_name)
        vec = st.encode("test")
        import numpy as _np
        dim = int((_np.asarray(vec)).shape[-1]) if hasattr(_np.asarray(vec), 'shape') else (len(vec) if hasattr(vec, '__len__') else 384)
        print(f"   ✓ 取得完了（次元数: {dim}）")
    except Exception as e:
        print(f"   - STモデル取得エラー: {e}")

    # PANNs ウォームアップ（既定で実施。必要なら DISABLE_PANNS=1 で無効化）
    print("\n4. PANNs(AudioSet) ウォームアップ...")
    disable_panns = str(os.getenv('DISABLE_PANNS') or '').lower() in ('1', 'true')
    if disable_panns:
        print("   SKIP: DISABLE_PANNS=1 が設定されています")
    else:
        try:
            from panns_inference import AudioTagging
            print("   panns_inference をロード中...")
            # インスタンス生成時にチェックポイントがダウンロードされるため、推論は行わない
            _ = AudioTagging(checkpoint_path=None, device='cpu')
            print("   ✓ PANNs のチェックポイントを取得しました（推論スキップ）")
        except Exception as e:
            print(f"   - PANNs 取得でエラー: {e}")

    print("\n5. セットアップ完了！")
    print("   サーバーを起動してください: npm start")
    print("=" * 60)

if __name__ == "__main__":
    download_models()
