# cli.py

import json
import os
import sys

import requests
import typer
from dotenv import load_dotenv

from app.utils import convert_masking_response_to_decode_request
from gpt_handler import GPTHandler


app = typer.Typer()
DEFAULT_CATEGORIES = ["ORG", "PERSON", "LOCATION", "POSITION"]
AVAILABLE_CATEGORIES = [
	"ORG",
	"PERSON",
	"LOCATION",
	"POSITION",
	"DATE",
	"EVENT",
	"PRODUCT",
	"NORP",
	"FACILITY",
	"GPE",
	"LAW",
	"LANGUAGE",
	"MONEY",
	"PERCENT",
	"TIME",
	"QUANTITY",
	"ORDINAL",
	"CARDINAL",
	"EMAIL",
	"PHONE",
	"PROJECT",
	"DEPARTMENT",
]

# .envファイルから環境変数を読み込み
load_dotenv()


def mask_text(
	text: str,
	categories: list[str] = None,
	key_values_to_mask: dict[str, str] = None,
	values_to_mask: list[str] = None,
) -> dict:
	"""
	テキストをマスキング処理します（指定されたカテゴリに基づく）。
	"""
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
		raise typer.Exit(f"マスキングエラー: {response.status_code} - {response.text}")


def decode_text(masking_result: dict) -> str:
	"""
	マスキングされたテキストをデコードします。
	"""
	# マスキング結果をデコードリクエスト形式に変換
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
		raise typer.Exit(f"デコードエラー: {response.status_code} - {response.text}")


@app.command()
def process(
	text: str = typer.Argument(
		None,
		help=("処理する日本語テキストです。指定しない場合は標準入力から読み取ります。"),
	),
	categories: list[str] = typer.Option(  # noqa: B008
		None,
		"--category",
		"-c",
		help="マスキングするカテゴリのリストです。",
	),
	key_values_file: str = typer.Option(
		None,
		"--key-values-file",
		"-k",
		help="key_values_to_maskを指定するJSONファイルのパスです。",
	),
	values_file: str = typer.Option(
		None,
		"--values-file",
		"-v",
		help="values_to_maskを指定するJSONファイルのパスです。",
	),
):
	"""
	ローカルマスキングサービスとGPTを使用して日本語テキストをマスキングおよびデコードします。
	"""
	# 入力テキストの取得
	if text:
		input_text = text
	else:
		typer.echo(
			"処理する日本語テキストを入力してください（終了するには Ctrl+D または "
			"Ctrl+Z を押して Enter）。"
		)

		try:
			input_text = sys.stdin.read()
		except KeyboardInterrupt:
			raise typer.Exit("入力がキャンセルされました。") from None

	# カテゴリの確認と選択
	if categories is None:
		typer.secho("\nマスキング可能なカテゴリ一覧:", fg=typer.colors.BLUE, bold=True)
		for idx, category in enumerate(AVAILABLE_CATEGORIES, start=1):
			typer.echo(f"{idx}. {category}")

		typer.echo(
			"\nマスキングしたいカテゴリに対応する番号をカンマ区切りで入力してください。"
		)
		typer.echo("例：ORGとPERSONを選択する場合は 1,2 と入力します。")
		typer.echo("デフォルトカテゴリを使用するには Enter キーを押してください。")

		user_input = typer.prompt("選択", default="")

		if user_input.strip() == "":
			selected_categories = DEFAULT_CATEGORIES
			typer.secho("\nデフォルトのカテゴリを使用します:", fg=typer.colors.YELLOW, bold=True)
			typer.echo(", ".join(selected_categories))
		else:
			try:
				selected_indices = [
					int(i.strip()) for i in user_input.split(",") if i.strip().isdigit()
				]
				selected_categories = [
					AVAILABLE_CATEGORIES[i - 1]
					for i in selected_indices
					if 1 <= i <= len(AVAILABLE_CATEGORIES)
				]
				if not selected_categories:
					typer.secho(
						"有効なカテゴリが選択されませんでした。デフォルトのカテゴリを使用します。",
						fg=typer.colors.RED,
						bold=True,
					)
					selected_categories = DEFAULT_CATEGORIES
				else:
					typer.secho("\n選択したカテゴリ:", fg=typer.colors.YELLOW, bold=True)
					typer.echo(", ".join(selected_categories))
			except Exception as e:
				typer.secho(
					f"無効な入力です。デフォルトのカテゴリを使用します。エラー: {e}",
					fg=typer.colors.RED,
					bold=True,
				)
				selected_categories = DEFAULT_CATEGORIES
	else:
		# 指定されたカテゴリの検証
		invalid_categories = [cat for cat in categories if cat not in AVAILABLE_CATEGORIES]
		if invalid_categories:
			typer.secho(
				f"無効なカテゴリが指定されました: {', '.join(invalid_categories)}",
				fg=typer.colors.RED,
				bold=True,
			)
			raise typer.Exit(code=1)
		selected_categories = categories
		typer.secho("\n指定されたカテゴリを使用します:", fg=typer.colors.YELLOW, bold=True)
		typer.echo(", ".join(selected_categories))

	# key_values_to_maskとvalues_to_maskの読み込み
	key_values_to_mask = {"株式会社Lightblue": "lead tech"}
	values_to_mask = ["RAG Ready診断", "最先端アルゴリズム"]

	if key_values_file:
		try:
			with open(key_values_file, encoding="utf-8") as f:
				key_values_to_mask = json.load(f)
		except Exception as e:
			typer.secho(f"key_values_fileの読み込みエラー: {e}", fg=typer.colors.RED)
			raise typer.Exit(code=1) from e

	if values_file:
		try:
			with open(values_file, encoding="utf-8") as f:
				values_to_mask = json.load(f)
		except Exception as e:
			typer.secho(f"values_fileの読み込みエラー: {e}", fg=typer.colors.RED)
			raise typer.Exit(code=1) from e

	try:
		typer.secho("\n元のテキスト:", fg=typer.colors.GREEN, bold=True)
		typer.echo(input_text)

		# 1. テキストのマスキング
		masking_result = mask_text(
			text=input_text,
			categories=selected_categories,
			key_values_to_mask=key_values_to_mask,
			values_to_mask=values_to_mask,
		)

		typer.secho("\nマスキングされたテキスト:", fg=typer.colors.GREEN, bold=True)
		typer.echo(masking_result["masked_text"])

		# 2. GPTとの対話
		openai_api_key = os.getenv("OPENAI_API_KEY")
		if not openai_api_key:
			raise typer.Exit("エラー: 環境変数に OPENAI_API_KEY が設定されていません。")

		gpt = GPTHandler(openai_api_key)
		messages = [
			{"role": "system", "content": "あなたは要約や分析を行うアシスタントです。"},
			{
				"role": "user",
				"content": (
					"以下のテキストを3行で要約してください：\n\n" f"{masking_result['masked_text']}"
				),
			},
		]

		gpt_response = gpt.ask(messages)

		typer.secho("\nGPTの応答（マスキング済み）:", fg=typer.colors.GREEN, bold=True)
		typer.echo(gpt_response)

		# 3. GPTの応答をデコード
		gpt_masking_result = {
			"masked_text": gpt_response,
			"entity_mapping": masking_result["entity_mapping"],
		}
		decoded_response = decode_text(gpt_masking_result)

		typer.secho("\nデコードされた応答:", fg=typer.colors.GREEN, bold=True)
		typer.echo(decoded_response)

	except Exception as e:
		typer.secho(f"エラー: {str(e)}", fg=typer.colors.RED, err=True)
		raise typer.Exit(code=1) from e


if __name__ == "__main__":
	app()
