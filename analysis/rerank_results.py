#!/usr/bin/env python3
# coding: utf-8
"""
rerank_results.py
- Cross-Encoderを使用して検索結果を再ランキング
- より高精度な意味的類似度判定
"""

import sys
import json
import os
from pathlib import Path
from typing import List, Dict

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

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Query and results JSON required"}))
        sys.exit(1)
    
    query = sys.argv[1]
    try:
        results = json.loads(sys.argv[2])
    except json.JSONDecodeError:
        print(json.dumps({"error": "Invalid JSON in results"}))
        sys.exit(1)
    
    try:
        # .envファイルを読み込み
        load_env()
        
        # Cross-Encoderモデルを使用
        from sentence_transformers import CrossEncoder
        
        # 環境変数でモデルを上書き可能（デフォルトは日本語特化）
        model_name = os.getenv('RERANK_MODEL', 'hotchpotch/japanese-reranker-cross-encoder-base-v1')
        model = CrossEncoder(model_name)
        
        # クエリと各結果のペアを作成
        pairs = []
        for result in results:
            # キャプションとタグを結合してテキスト作成
            text = result.get('caption', '')
            tags = result.get('tags', [])
            if isinstance(tags, list):
                text += ' ' + ' '.join(tags)
            
            pairs.append([query, text])
        
        # 再ランキングスコアを計算
        if pairs:
            scores = model.predict(pairs)
            
            print(f"[DEBUG] Reranking {len(pairs)} pairs for query: '{query}'", file=sys.stderr)
            
            # スコアと元結果を結合
            for i, result in enumerate(results):
                result['rerank_score'] = float(scores[i])
                # デバッグ情報を出力
                caption = result.get('caption', '')[:50] + '...' if len(result.get('caption', '')) > 50 else result.get('caption', '')
                tags = result.get('tags', [])[:5]  # 最初の5つのタグ
                print(f"[DEBUG] Result {i+1}: Score={scores[i]:.3f}, Caption='{caption}', Tags={tags}", file=sys.stderr)
            
            print("[DEBUG] Before rerank sort:", file=sys.stderr)
            for i, result in enumerate(results[:3]):  # 上位3件のみ表示
                caption = result.get('caption', '')[:30] + '...' if len(result.get('caption', '')) > 30 else result.get('caption', '')
                print(f"  {i+1}. Score={result.get('rerank_score', 0):.3f}, Caption='{caption}'", file=sys.stderr)
            
            # スコア順でソート（降順）
            results.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
            
            print("[DEBUG] After rerank sort:", file=sys.stderr)
            for i, result in enumerate(results[:3]):  # 上位3件のみ表示
                caption = result.get('caption', '')[:30] + '...' if len(result.get('caption', '')) > 30 else result.get('caption', '')
                print(f"  {i+1}. Score={result.get('rerank_score', 0):.3f}, Caption='{caption}'", file=sys.stderr)
        
        # 結果を返す
        print(json.dumps({
            "success": True,
            "results": results,
            "reranked": True
        }))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
