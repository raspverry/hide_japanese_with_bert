# client.py
import os
import sys

import requests
import typer
from dotenv import load_dotenv

from app.utils import convert_masking_response_to_decode_request
from gpt_handler import GPTHandler


app = typer.Typer()
DEFAULT_CATEGORIES = ["ORG", "PERSON", "LOCATION", "POSITION"]

# Load environment variables from .env file
load_dotenv()


def mask_text(text: str, categories: list[str] = None) -> dict:
	"""
	テキストをマスキング処理 (Mask the text based on specified categories)
	"""
	response = requests.post(
		"http://localhost:8000/mask_text",
		headers={"Content-Type": "application/json"},
		json={
			"text": text,
			"categories_to_mask": categories or [],
			"mask_style": "descriptive",
		},
	)

	if response.status_code == 200:
		return response.json()
	else:
		raise typer.Exit(f"Masking Error: {response.status_code} - {response.text}")


def decode_text(masking_result: dict) -> str:
	"""
	マスキングされたテキストをデコード (Decode the masked text)
	"""
	# Convert masking result to decode request format
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
		raise typer.Exit(f"Decoding Error: {response.status_code} - {response.text}")


@app.command()
def process(
	text: str | None = typer.Argument(
		None,
		help=(
        "Japanese text to process. If not provided, the script will read "
        "from standard input."
    ),
	),
	categories: list[str] = typer.Option(  # noqa: B008
        DEFAULT_CATEGORIES,
        "-c",
        "--categories",
        help="Categories to mask (default: ORG PERSON LOCATION POSITION).",
    ),
):
	"""
	Mask and decode Japanese text using local masking services and GPT.
	"""
	# Determine the input text source
	if text:
		input_text = text
	else:
		typer.echo(
			"Please enter the Japanese text (press Ctrl+D or Ctrl+Z then Enter to end input):"
		)
		try:
			input_text = sys.stdin.read()
		except KeyboardInterrupt:
			raise typer.Exit("Input cancelled.") from None

	try:
		typer.secho("\nOriginal Text:", fg=typer.colors.GREEN, bold=True)
		typer.echo(input_text)

		# 1. Mask the text
		masking_result = mask_text(text=input_text, categories=categories)

		typer.secho("\nMasked Text:", fg=typer.colors.GREEN, bold=True)
		typer.echo(masking_result["masked_text"])

		# 2. Interact with GPT
		openai_api_key = os.getenv("OPENAI_API_KEY")
		if not openai_api_key:
			raise typer.Exit("Error: OPENAI_API_KEY is not set in the environment variables.")

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

		typer.secho("\nGPT Response (Masked):", fg=typer.colors.GREEN, bold=True)
		typer.echo(gpt_response)

		# 3. Decode GPT's response
		gpt_masking_result = {
			"masked_text": gpt_response,
			"entity_mapping": masking_result["entity_mapping"],
		}
		decoded_response = decode_text(gpt_masking_result)

		typer.secho("\nDecoded Response:", fg=typer.colors.GREEN, bold=True)
		typer.echo(decoded_response)

	except Exception as e:
		typer.secho(f"Error: {str(e)}", fg=typer.colors.RED, err=True)
		raise typer.Exit(code=1) from e


if __name__ == "__main__":
	app()
