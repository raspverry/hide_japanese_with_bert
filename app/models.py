# app/models.py

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from dataclasses import dataclass

class EnhancedMaskingRequest(BaseModel):
    """マスキングリクエストのモデル"""
    text: str = Field(..., max_length=5000, description="マスキング対象のテキスト")
    categories_to_mask: Optional[List[str]] = Field(
        None,
        description="マスキングするカテゴリのリスト"
    )
    mask_style: Optional[str] = Field(
        "descriptive",
        description='"descriptive" または "simple" のマスキングスタイル'
    )

class Position(BaseModel):
    """位置情報のモデル"""
    start: int
    end: int

class DetectedEntity(BaseModel):
    """検出されたエンティティのモデル"""
    original: str
    category: str
    mask_token: str
    position: Position
    source: str

class DebugInfo(BaseModel):
    """デバッグ情報のモデル"""
    detected_entities: List[DetectedEntity]

class MaskingResponse(BaseModel):
    """マスキングレスポンスのモデル"""
    masked_text: str
    entity_mapping: Dict[str, Dict[str, str]]
    debug_info: DebugInfo

class DecodeRequest(BaseModel):
    """デコードリクエストのモデル"""
    masked_text: str = Field(..., description="マスキングされたテキスト")
    entity_mapping: Dict[str, Dict[str, str]] = Field(..., description="マスキングトークンと元のテキストのマッピング")

class DecodeResponse(BaseModel):
    """デコードレスポンスのモデル"""
    decoded_text: str

@dataclass
class Entity:
    """エンティティ情報を保持するデータクラス"""
    text: str
    category: str
    start: int
    end: int
    priority: int = 0
    source: str = "rule"  # "rule" または "ginza"
