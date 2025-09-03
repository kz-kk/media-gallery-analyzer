#!/usr/bin/env python3
# coding: utf-8
"""
secure_logging.py
- セキュリティに配慮したロギングユーティリティ
- 認証情報やセンシティブな情報をマスクする
"""
import re
import json
from typing import Any, Dict, List, Union

# センシティブなキーのパターン
SENSITIVE_KEY_PATTERNS = [
    r'.*key.*',
    r'.*token.*',
    r'.*secret.*',
    r'.*password.*',
    r'.*pwd.*',
    r'.*auth.*',
    r'.*api[-_]?key.*',
    r'.*access[-_]?token.*',
    r'.*bearer.*',
]

# センシティブなヘッダー
SENSITIVE_HEADERS = {
    'authorization',
    'x-api-key',
    'x-auth-token',
    'cookie',
    'set-cookie'
}

def is_sensitive_key(key: str) -> bool:
    """キーがセンシティブかどうかを判定"""
    key_lower = key.lower()
    for pattern in SENSITIVE_KEY_PATTERNS:
        if re.match(pattern, key_lower):
            return True
    return key_lower in SENSITIVE_HEADERS

def mask_value(value: str, show_chars: int = 4) -> str:
    """値をマスクする（最初と最後の数文字のみ表示）"""
    if not value or len(value) <= show_chars * 2:
        return "***"
    
    # Bearer トークンの特殊処理
    if value.startswith('Bearer '):
        token = value[7:]
        if len(token) > show_chars * 2:
            masked_token = f"{token[:show_chars]}...{token[-show_chars:]}"
        else:
            masked_token = "***"
        return f"Bearer {masked_token}"
    
    return f"{value[:show_chars]}...{value[-show_chars:]}"

def sanitize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """辞書内のセンシティブな値をマスク"""
    if not isinstance(data, dict):
        return data
    
    sanitized = {}
    for key, value in data.items():
        if is_sensitive_key(key):
            if isinstance(value, str):
                sanitized[key] = mask_value(value)
            else:
                sanitized[key] = "***"
        elif isinstance(value, dict):
            sanitized[key] = sanitize_dict(value)
        elif isinstance(value, list):
            sanitized[key] = sanitize_list(value)
        else:
            sanitized[key] = value
    return sanitized

def sanitize_list(data: List[Any]) -> List[Any]:
    """リスト内のセンシティブな値をマスク"""
    sanitized = []
    for item in data:
        if isinstance(item, dict):
            sanitized.append(sanitize_dict(item))
        elif isinstance(item, list):
            sanitized.append(sanitize_list(item))
        else:
            sanitized.append(item)
    return sanitized

def safe_debug_print(message: str, data: Any = None):
    """セキュアなデバッグ出力"""
    if data is None:
        print(f"[DEBUG] {message}")
        return
    
    if isinstance(data, dict):
        sanitized_data = sanitize_dict(data)
    elif isinstance(data, list):
        sanitized_data = sanitize_list(data)
    else:
        # URLやその他の文字列からセンシティブな情報を削除
        sanitized_data = sanitize_url(str(data))
    
    print(f"[DEBUG] {message}: {sanitized_data}")

def sanitize_url(url: str) -> str:
    """URLからセンシティブなクエリパラメータをマスク"""
    # APIキーなどがURLに含まれている場合の処理
    patterns = [
        (r'(api[-_]?key=)([^&]+)', r'\1***'),
        (r'(token=)([^&]+)', r'\1***'),
        (r'(auth=)([^&]+)', r'\1***'),
        (r'(Bearer\s+)(\S+)', lambda m: f"{m.group(1)}{mask_value(m.group(2))}"),
    ]
    
    sanitized = url
    for pattern, replacement in patterns:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    
    return sanitized

def safe_json_dumps(data: Any, **kwargs) -> str:
    """センシティブな情報をマスクしてJSON出力"""
    if isinstance(data, dict):
        sanitized = sanitize_dict(data)
    elif isinstance(data, list):
        sanitized = sanitize_list(data)
    else:
        sanitized = data
    
    return json.dumps(sanitized, **kwargs)

# 環境変数の安全な取得
def get_safe_env(key: str, default: str = None, mask_in_logs: bool = True) -> str:
    """環境変数を安全に取得（ログ出力時にマスク）"""
    import os
    value = os.getenv(key, default)
    
    if mask_in_logs and is_sensitive_key(key) and value:
        # この値がログに出力される際は自動的にマスクされるようにマーカーを付ける
        # 実際の値は返すが、ログ出力時の識別用
        setattr(value, '_sensitive', True)
    
    return value

if __name__ == "__main__":
    # テスト
    test_data = {
        "api_key": "sk-1234567890abcdef",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
        "normal_data": "This is normal data",
        "nested": {
            "password": "secret123",
            "username": "user123"
        }
    }
    
    print("=== セキュアロギングのテスト ===")
    print("\n元のデータ:")
    print(json.dumps(test_data, indent=2))
    
    print("\nサニタイズ後:")
    print(safe_json_dumps(test_data, indent=2))
    
    print("\nヘッダーのテスト:")
    headers = {"Authorization": f"Bearer {'x' * 50}", "Content-Type": "application/json"}
    safe_debug_print("Headers", headers)