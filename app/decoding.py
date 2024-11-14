# app/decoding.py

import re
from typing import Dict
import structlog

from app.models import DecodeResponse  # DecodeRequest は使用していないため削除

logger = structlog.getLogger(__name__)

class EnhancedTextDecoder:
    """マスキングされたテキストを元に戻すデコーダークラス"""

    def decode_text(self, masked_text: str, entity_mapping: Dict[str, Dict[str, str]]) -> str:
        """マスキングされたテキストを元のテキストに復元します"""
        # マスキングトークンの長さで降順ソートして、誤置換を防ぎます
        sorted_tokens = sorted(entity_mapping.keys(), key=len, reverse=True)

        decoded_text = masked_text

        for token in sorted_tokens:
            # マスキングトークンが変形する可能性に備えて、正規表現を使用
            pattern = re.escape(token)
            replacement = entity_mapping[token]['text']
            decoded_text = re.sub(pattern, replacement, decoded_text)

            logger.debug(f"デコード適用: {token} -> {replacement}")

        return decoded_text
