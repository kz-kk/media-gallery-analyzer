#!/usr/bin/env python3
"""
必要なモデルをダウンロードするセットアップスクリプト
"""
import os
import sys
from pathlib import Path

def download_models():
    print("=" * 60)
    print("Media Gallery Analyzer - モデルセットアップ")
    print("=" * 60)
    
    # Plamoモデルのダウンロード（Hugging Faceキャッシュを使用）
    print("\n1. Plamo埋め込みモデルを確認中...")
    try:
        from transformers import AutoTokenizer, AutoModel
        
        model_name = "pfnet/plamo-embedding-1b"
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        
        # キャッシュを確認
        cached = False
        for item in hf_cache.glob("models--pfnet--plamo-embedding-1b*"):
            if item.is_dir():
                cached = True
                print(f"   ✓ 既にキャッシュ済み: {model_name}")
                break
        
        if not cached:
            print(f"   ダウンロード中: {model_name}")
            print("   初回ダウンロードには時間がかかります（約1.8GB）...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
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
    
    print("\n3. セットアップ完了！")
    print("   サーバーを起動してください: npm start")
    print("=" * 60)

if __name__ == "__main__":
    download_models()