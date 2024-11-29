# app/models.py

from dataclasses import dataclass

from pydantic import BaseModel, Field


class EnhancedMaskingRequest(BaseModel):
	"""マスキングリクエストのモデル"""

	text: str = Field(..., max_length=5000, description="マスキング対象のテキスト")
	categories_to_mask: list[str] | None = Field(
		None, description="マスキングするカテゴリのリスト"
	)
	mask_style: str | None = Field(
		"descriptive", description='"descriptive" または "simple" のマスキングスタイル'
	)
	key_values_to_mask: dict[str, str] | None = Field(
		None, description="キーと値のペアで指定されたマスキングルール"
	)
	values_to_mask: list[str] | None = Field(
		None, description="UUIDでマスキングする値のリスト"
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

	detected_entities: list[DetectedEntity]


class MaskingResponse(BaseModel):
	"""マスキングレスポンスのモデル"""

	masked_text: str
	entity_mapping: dict[str, dict[str, str]]
	debug_info: DebugInfo


class DecodeRequest(BaseModel):
	"""デコードリクエストのモデル"""

	masked_text: str = Field(..., description="マスキングされたテキスト")
	entity_mapping: dict[str, dict[str, str]] = Field(
		..., description="マスキングトークンと元のテキストのマッピング"
	)


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
