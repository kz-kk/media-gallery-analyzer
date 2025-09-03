#!/usr/bin/env python3
# coding: utf-8
"""
get_embedding.py
- テキストからエンベディングベクトルを生成してJSON出力
- server.jsのセマンティック検索で使用
"""

import sys
import json
import os
import numpy as np
from pathlib import Path

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
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Query text required"}))
        sys.exit(1)
    
    query_text = sys.argv[1]
    
    try:
        # .envファイルを読み込み
        load_env()
        
        # モデルは常に Plamo を使用
        emb_model_name = os.getenv("EMB_MODEL", "pfnet/plamo-embedding-1b")
        import torch
        from transformers import AutoTokenizer, AutoModel

        trusted_models = [
            "pfnet/plamo-embedding-1b"
        ]
        trust_code = emb_model_name in trusted_models
        if not trust_code:
            print(f"[SECURITY WARNING] Untrusted model: {emb_model_name}. Using with trust_remote_code=False", file=sys.stderr)

        tokenizer = AutoTokenizer.from_pretrained(emb_model_name, trust_remote_code=trust_code)
        model = AutoModel.from_pretrained(emb_model_name, trust_remote_code=trust_code)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        with torch.inference_mode():
            # 検索クエリには encode_query（バッチ）を使用
            embedding_tensor = model.encode_query([query_text], tokenizer)
            embedding_np = embedding_tensor.cpu().numpy()
            # 先頭要素を1次元に
            if getattr(embedding_np, 'ndim', 1) > 1:
                embedding = embedding_np[0]
            else:
                embedding = embedding_np
        
        # 正規化（Cosine用の安定化）
        try:
            import numpy as _np
            nrm = float(_np.linalg.norm(embedding)) or 1.0
            embedding = (embedding / nrm).tolist()
        except Exception:
            embedding = embedding.tolist() if hasattr(embedding, 'tolist') else embedding

        # JSON出力
        result = {
            "embedding": embedding,
            "model": emb_model_name,
            "dimension": len(embedding)
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
