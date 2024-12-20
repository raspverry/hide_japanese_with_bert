# client.py

import os

import requests
from dotenv import load_dotenv

from app.utils import convert_masking_response_to_decode_request
from gpt_handler import GPTHandler


load_dotenv()


def mask_text(
	text: str,
	categories: list[str] | None = None,
	key_values_to_mask: dict[str, str] | None = None,
	values_to_mask: list[str] | None = None,
) -> dict:
	"""テキストをマスキング処理"""
	response = requests.post(
		"http://localhost:8000/mask_text",
		headers={"Content-Type": "application/json"},
		json={
			"text": text,
			"categories_to_mask": categories or [],
			"mask_style": "descriptive",
			"key_values_to_mask": key_values_to_mask or {},
			"values_to_mask": values_to_mask or [],
		},
	)

	if response.status_code == 200:
		return response.json()
	else:
		raise Exception(f"Masking Error: {response.status_code} - {response.text}")


def decode_text(masking_result: dict) -> str:
	"""マスキングされたテキストをデコード"""
	# マスキング結果からデコードリクエストを生成
	decode_request = convert_masking_response_to_decode_request(masking_result)

	response = requests.post(
		"http://localhost:8000/decode_text",
		headers={"Content-Type": "application/json"},
		json={
			"masked_text": decode_request.masked_text,
			"entity_mapping": decode_request.entity_mapping,
		},
	)

	if response.status_code == 200:
		return response.json()["decoded_text"]
	else:
		raise Exception(f"Decoding Error: {response.status_code} - {response.text}")


if __name__ == "__main__":
	# テストテキスト
	text = """
    最先端アルゴリズムの社会実装に取り組むAIスタートアップ、株式会社Lightblue(代表取締役:園田亜斗夢、本社:東京都千代田区、以下「Lightblue」)は、
    生成AIの導入効果を最大化するための診断サービス「RAG Ready診断」をリリースいたしました。
    """

	try:
		print("Original Text:")
		print(text)

		# 1. テキストのマスキング
		key_values_to_mask = {"株式会社Lightblue": "lead tech"}
		values_to_mask = ["RAG Ready診断", "最先端アルゴリズム"]

		masking_result = mask_text(
			text=text,
			categories=["ORG", "PERSON", "LOCATION", "POSITION"],
			key_values_to_mask=key_values_to_mask,
			values_to_mask=values_to_mask,
		)

		print("Masked Text:")
		print(masking_result["masked_text"])

		# 2. GPTに質問
		key = os.getenv("OPENAI_API_KEY")
		gpt = GPTHandler(key)
		messages = [
			{"role": "system", "content": "あなたは要約や分析を行うアシスタントです。"},
			{
				"role": "user",
				"content": (
					"以下のテキストを3行で要約してください：\n\n"
					f"{masking_result['masked_text']}"
				),
			},
		]

		gpt_response = gpt.ask(messages)

		print("\nGPT Response (Masked):")
		print(gpt_response)

		# 3. GPTの応答をデコード
		# マスキング結果を新しい構造に変換
		gpt_masking_result = {
			"masked_text": gpt_response,
			"entity_mapping": masking_result["entity_mapping"],
		}
		print("\nGPT Masking Result:", gpt_masking_result)
		decoded_response = decode_text(gpt_masking_result)

		print("\nDecoded Response:")
		print(decoded_response)

	except Exception as e:
		print(f"Error: {str(e)}")
