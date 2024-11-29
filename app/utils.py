# app/utils.py


from app.models import DecodeRequest, MaskingResponse


def convert_masking_response_to_decode_request(
	masking_response: MaskingResponse | dict,
) -> DecodeRequest:
	"""
	マスキングレスポンスをデコードリクエスト形式に変換します。

	Args:
	masking_response: MaskingResponseオブジェクト
			または
			辞書形式のマスキングレスポンス

	Returns:
	DecodeRequest: デコードリクエストオブジェクト

	Example:
	```python
	# APIレスポンスとして受け取った辞書を変換
	masking_response = {
	"masked_text": "<<組織_1>>の<<人物_2>>",
	"entity_mapping": {
	"<<組織_1>>": {"text":"株式会社A","category":"ORG","source":"rule"},
	"<<人物_2>>":{"text":"山田太郎","category":"PERSON","source":"ginza"}
	},
	"debug_info": {...}
	}
	decode_request = convert_masking_response_to_decode_request(masking_response)

	# MaskingResponseオブジェクトを変換
	masking_response_obj = MaskingResponse(...)
	decode_request=convert_masking_response_to_decode_request(masking_response_obj)
	```
	"""
	# 入力が辞書の場合（APIレスポンスなど）
	if isinstance(masking_response, dict):
		return DecodeRequest(
			masked_text=masking_response["masked_text"],
			entity_mapping=masking_response["entity_mapping"],
		)

	# 入力がMaskingResponseオブジェクトの場合
	elif isinstance(masking_response, MaskingResponse):
		return DecodeRequest(
			masked_text=masking_response.masked_text,
			entity_mapping=masking_response.entity_mapping,
		)

	else:
		raise ValueError(
			"入力はMaskingResponseオブジェクトまたは辞書である必要があります。"
		)


def format_decode_request_to_json(decode_request: DecodeRequest) -> str:
	"""
	DecodeRequestオブジェクトをcurlコマンドで使用可能なJSON文字列に変換します。

	Args:
	decode_request: DecodeRequestオブジェクト

	Returns:
	str: JSON形式の文字列

	Example:
	```python
	decode_request = DecodeRequest(...)
	json_str = format_decode_request_to_json(decode_request)
	print(f"curl -X POST 'http://localhost:8000/decode_text' "
	f"-H 'Content-Type: application/json' -d '{json_str}'")
	```
	"""
	import json

	request_dict = {
		"masked_text": decode_request.masked_text,
		"entity_mapping": decode_request.entity_mapping,
	}

	# 日本語が正しく表示されるようにensure_ascii=Falseを設定
	return json.dumps(request_dict, ensure_ascii=False)


def get_curl_command(
	decode_request: DecodeRequest, endpoint: str = "http://localhost:8000/decode_text"
) -> str:
	"""
	DecodeRequestオブジェクトからcurlコマンドを生成します。

	Args:
	decode_request: DecodeRequestオブジェクト
	endpoint: APIエンドポイントURL（デフォルト: "http://localhost:8000/decode_text"）

	Returns:
	str: curlコマンド

	Example:
	```python
	decode_request = DecodeRequest(...)
	curl_command = get_curl_command(decode_request)
	print(curl_command)  # curlコマンドを出力
	```
	"""
	json_data = format_decode_request_to_json(decode_request)
	return (
		f"curl -X POST '{endpoint}' "
		f"-H 'Content-Type: application/json' "
		f"-d '{json_data}'"
	)
