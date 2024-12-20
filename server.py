# server.py

import os
import warnings

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException

# ロギング設定をインポート（設定スクリプトを実行）
import app.logger_config  # この行でロギング設定が適用されます
from app.decoding import EnhancedTextDecoder
from app.masking import EnhancedTextMasker
from app.models import (
	DebugInfo,
	DecodeRequest,
	DecodeResponse,
	EnhancedMaskingRequest,
	MaskingResponse,
)


# PyTorch 関連の警告を無視
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ロガーの取得
logger = structlog.get_logger(__name__)

app = FastAPI(title="高度なテキストマスキングAPI")


@app.post("/mask_text", response_model=MaskingResponse)
async def mask_text_endpoint(request: EnhancedMaskingRequest):
	"""テキストマスキングエンドポイント"""
	try:
		masker = EnhancedTextMasker()
		masked_text, entity_mapping, debug_info = masker.mask_text(
			text=request.text,
			categories=request.categories_to_mask,
			mask_style=request.mask_style,
			key_values_to_mask=request.key_values_to_mask,
			values_to_mask=request.values_to_mask,
		)

		logger.info(
			"mask_text",
			original_text=request.text,
			masked_text=masked_text,
			categories=request.categories_to_mask,
			mask_style=request.mask_style,
		)

		return MaskingResponse(
			masked_text=masked_text,
			entity_mapping=entity_mapping,
			debug_info=DebugInfo(detected_entities=debug_info),
		)

	except FileNotFoundError:
		logger.error("ルールファイルが見つかりません", rules_file=request.text)
		raise HTTPException(
			status_code=400, detail="ルールファイルが見つかりません。"
		) from None
	except Exception as e:
		logger.error("テキスト処理中にエラーが発生しました", error=str(e))
		raise HTTPException(
			status_code=500, detail="テキスト処理中に予期しないエラーが発生しました。"
		) from e


@app.post("/decode_text", response_model=DecodeResponse)
async def decode_text_endpoint(request: DecodeRequest):
	"""テキストデコードエンドポイント"""
	print("####")
	print(request)
	print("####")
	try:
		decoder = EnhancedTextDecoder()
		decoded_text = decoder.decode_text(request.masked_text, request.entity_mapping)

		logger.info(
			"decode_text", masked_text=request.masked_text, decoded_text=decoded_text
		)

		return DecodeResponse(decoded_text=decoded_text)

	except Exception as e:
		logger.error("デコード処理中にエラーが発生しました", error=str(e))
		raise HTTPException(
			status_code=500, detail="デコード処理中に予期しないエラーが発生しました。"
		) from e


if __name__ == "__main__":
	# APIサーバー起動
	rules_file_path = "masking_rules.json"
	if not os.path.exists(rules_file_path):
		logger.error("ルールファイルが見つかりません", rules_file=rules_file_path)
		exit(1)
	uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
