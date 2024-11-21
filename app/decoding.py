# app/decoding.py

import re

import structlog


logger = structlog.getLogger(__name__)


class EnhancedTextDecoder:
	"""マスキングされたテキストを元に戻すデコーダークラス"""

	def decode_text(
		self, masked_text: str, entity_mapping: dict[str, dict[str, str]]
	) -> str:
		"""マスキングされたテキストを元のテキストに復元します"""
		# マスキングトークンの長さで降順ソートして、誤置換を防ぎます
		sorted_entities = sorted(
			entity_mapping.values(), key=lambda x: len(x["masked_text"]), reverse=True
		)

		decoded_text = masked_text

		for entity in sorted_entities:
			masked = entity["masked_text"]
			original = entity["original_text"]
			# マスキングトークンが変形する可能性に備えて、正規表現を使用
			pattern = re.escape(masked)
			decoded_text = re.sub(pattern, original, decoded_text)

			logger.debug(f"デコード適用: {masked} -> {original}")

		return decoded_text
