import os

import gradio as gr
import requests
from dotenv import load_dotenv

from app.utils import convert_masking_response_to_decode_request
from gpt_handler import GPTHandler


# 環境変数をロード
load_dotenv()

# OpenAI APIキーの取得
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
	raise ValueError("環境変数 OPENAI_API_KEY が設定されていません。")

# マスキングサービスのエンドポイント
MASKING_ENDPOINT = "http://localhost:8000/mask_text"
DECODE_ENDPOINT = "http://localhost:8000/decode_text"

# デフォルトおよび利用可能なカテゴリ
DEFAULT_CATEGORIES: list[str] = []

# カテゴリの日本語表示用マッピング
CATEGORY_CODE_MAP = {
	"組織": "ORG",
	"人物": "PERSON",
	"場所": "LOCATION",
	"役職": "POSITION",
	"日付": "DATE",
	"イベント": "EVENT",
	"製品": "PRODUCT",
	"国籍/宗教/政治団体": "NORP",
	"施設": "FACILITY",
	"地政学的実体": "GPE",
	"法律": "LAW",
	"言語": "LANGUAGE",
	"金額": "MONEY",
	"割合": "PERCENT",
	"時間": "TIME",
	"数量": "QUANTITY",
	"序数": "ORDINAL",
	"基数": "CARDINAL",
	"メール": "EMAIL",
	"電話番号": "PHONE",
	"プロジェクト": "PROJECT",
	"部署": "DEPARTMENT",
}

# カテゴリ別の色マッピング
CATEGORY_COLOR_MAP = {
	"ORG": "rgba(255, 105, 180, 0.6)",  # 組織: ホットピンク
	"PERSON": "rgba(255, 165, 0, 0.6)",  # 人物: オレンジ
	"LOCATION": "rgba(50, 205, 50, 0.6)",  # 場所: ライムグリーン
	"POSITION": "rgba(30, 144, 255, 0.6)",  # 役職: ドジャーブルー
	"DATE": "rgba(147, 112, 219, 0.6)",  # 日付: パープル
	"EVENT": "rgba(255, 215, 0, 0.6)",  # イベント: ゴールド
	"PRODUCT": "rgba(220, 20, 60, 0.6)",  # 製品: クリムゾン
	"NORP": "rgba(70, 130, 180, 0.6)",  # 国籍/宗教/政治団体: スティールブルー
	"FACILITY": "rgba(34, 139, 34, 0.6)",  # 施設: フォレストグリーン
	"GPE": "rgba(244, 164, 96, 0.6)",  # 地政学的実体: サンドブラウン
	"LAW": "rgba(186, 85, 211, 0.6)",  # 法律: ミディアムパープル
	"LANGUAGE": "rgba(255, 140, 0, 0.6)",  # 言語: ダークオレンジ
	"MONEY": "rgba(46, 139, 87, 0.6)",  # 金額: シーグリーン
	"PERCENT": "rgba(65, 105, 225, 0.6)",  # 割合: ロイヤルブルー
	"TIME": "rgba(138, 43, 226, 0.6)",  # 時間: ブルーバイオレット
	"QUANTITY": "rgba(100, 149, 237, 0.6)",  # 数量: コーンフラワーブルー
	"ORDINAL": "rgba(219, 112, 147, 0.6)",  # 序数: パレオバイオレットレッド
	"CARDINAL": "rgba(218, 165, 32, 0.6)",  # 基数: ゴールデンロッド
	"EMAIL": "rgba(255, 20, 147, 0.6)",  # メール: ディープピンク
	"PHONE": "rgba(95, 158, 160, 0.6)",  # 電話番号: カデットブルー
	"PROJECT": "rgba(255, 127, 80, 0.6)",  # プロジェクト: コーラル
	"DEPARTMENT": "rgba(199, 21, 133, 0.6)",  # 部署: メディアムバイオレットレッド
}


def mask_text(text: str, categories: list[str] = None) -> dict:
	"""テキストをマスキング処理する関数"""
	response = requests.post(
		MASKING_ENDPOINT,
		headers={"Content-Type": "application/json"},
		json={
			"text": text,
			"categories_to_mask": categories or [],
			"mask_style": "descriptive",
		},
	)
	return (
		response.json()
		if response.status_code == 200
		else {"error": f"マスキングエラー: {response.status_code} - {response.text}"}
	)


def decode_text(masking_response: dict) -> str:
	"""マスキングされたテキストをデコードする関数"""
	try:
		decode_request = convert_masking_response_to_decode_request(masking_response)
	except ValueError as e:
		return f"デコードリクエスト変換エラー: {str(e)}"

	response = requests.post(
		DECODE_ENDPOINT,
		headers={"Content-Type": "application/json"},
		json=decode_request.dict(),
	)
	return (
		response.json().get("decoded_text", "")
		if response.status_code == 200
		else f"デコードエラー: {response.status_code} - {response.text}"
	)


def gpt_ask(masked_text: str) -> str:
	"""GPTにテキストを送信して応答を受け取る関数"""
	gpt = GPTHandler(OPENAI_API_KEY)
	messages = [
		{"role": "system", "content": "あなたは要約や分析を行うアシスタントです。"},
		{
			"role": "user",
			"content": f"以下のテキストを3行で要約してください：\n\n{masked_text}",
		},
	]
	return gpt.ask(messages)


def highlight_differences(
	original_text: str, masking_result: dict, highlight_color: str = None
) -> tuple:
	"""
	マスキング結果を元に、変更された部分をハイライトする関数

	Args:
		original_text (str): 元のテキスト
	masking_result (dict): マスキング処理の結果
		highlight_color (str): デフォルトのハイライト色（カテゴリ別の色を使用する場合は無視）

	Returns:
		tuple: (ハイライトされた元のテキスト, ハイライトされたマスキングテキスト)
	"""
	if "entity_mapping" not in masking_result:
		return original_text, masking_result.get("masked_text", original_text)

	# 元のテキストとマスキングされたテキストを準備
	masked_text = masking_result["masked_text"]

	# ハイライト位置を保存するリスト
	highlights_original = []
	highlights_masked = []

	# エンティティを長さの降順でソート
	entities = sorted(
		[(token, info) for token, info in masking_result["entity_mapping"].items()],
		key=lambda x: len(x[1]["text"]),
		reverse=True,
	)

	# 各エンティティの位置を特定
	for mask_token, info in entities:
		original_txt = info["text"]
		category = info["category"]
		color = CATEGORY_COLOR_MAP.get(category, highlight_color or "yellow")

		index = 0
		# 元のテキストでの位置を特定
		while True:
			pos = original_text.find(original_txt, index)
			if pos == -1:
				break

			# 既存のハイライトと重複していないか確認
			if not any(s <= pos < e for s, e, _, _ in highlights_original):
				highlights_original.append((pos, pos + len(original_txt), original_txt, color))
			index = pos + 1

		# マスクされたテキストでの位置を特定
		index = 0
		while True:
			pos = masked_text.find(mask_token, index)
			if pos == -1:
				break

			# 既存のハイライトと重複していないか確認
			if not any(s <= pos < e for s, e, _, _ in highlights_masked):
				highlights_masked.append((pos, pos + len(mask_token), mask_token, color))
			index = pos + 1

	# 位置でソート（後ろから処理）
	highlights_original.sort(reverse=True)
	highlights_masked.sort(reverse=True)

	# ハイライトを適用
	result_original = original_text
	for start, end, text, color in highlights_original:
		result_original = (
			result_original[:start]
			+ f'<span style="background-color: {color};">{text}</span>'
			+ result_original[end:]
		)

	result_masked = masked_text
	for start, end, text, color in highlights_masked:
		result_masked = (
			result_masked[:start]
			+ f'<span style="background-color: {color};">{text}</span>'
			+ result_masked[end:]
		)

	return result_original, result_masked


def process_text(input_text: str, categories: list[str]) -> dict:
	"""テキスト処理の全体プロセスを実行する関数"""
	if not input_text.strip():
		return {"error": "入力テキストが空です。"}

	# マスキング処理
	masking_result = mask_text(input_text, categories)
	if "error" in masking_result:
		return {"error": masking_result["error"]}

	# GPT要約
	gpt_response = gpt_ask(masking_result["masked_text"])

	# GPT応答の中のマスクトークンを検出してデコード用のマッピングを作成
	gpt_mapping = {}
	for mask_token, info in masking_result["entity_mapping"].items():
		if mask_token in gpt_response:
			gpt_mapping[mask_token] = info

	# GPT応答のマスクトークンをデコード
	decoded_response = gpt_response
	for mask_token, info in gpt_mapping.items():
		decoded_response = decoded_response.replace(mask_token, info["text"])

	# 原文とマスク文のハイライト処理
	highlighted_original, highlighted_masked = highlight_differences(
		input_text, masking_result, "rgba(255, 255, 0, 0.5)"
	)

	# GPT応答用のマッピング作成
	gpt_result_mapping = {
		"masked_text": gpt_response,
		"entity_mapping": gpt_mapping,
	}

	# GPT応答とデコード結果のハイライト処理
	highlighted_decoded, highlighted_gpt = highlight_differences(
		decoded_response, gpt_result_mapping, "rgba(144, 238, 144, 0.5)"
	)

	return {
		"original": highlighted_original,
		"masked": highlighted_masked,
		"gpt_response": highlighted_gpt,
		"decoded": highlighted_decoded,
	}


# Gradio インターフェースの作成
with gr.Blocks(
	theme=gr.themes.Soft(
		primary_hue="blue",
		secondary_hue="gray",
	)
) as demo:
	gr.Markdown(
		"""
        # テキストマスキング & 要約システム
        
        テキストの匿名化と要約を行うシステムです。
        """
	)

	with gr.Row():
		# 左側のカラム（入力部分）
		with gr.Column(scale=1):
			input_text = gr.Textbox(
				label="入力テキスト",
				placeholder="ここに日本語テキストを入力してください...",
				lines=10,
			)
			categories = gr.CheckboxGroup(
				label="マスキングカテゴリ",
				choices=[(cat, CATEGORY_CODE_MAP.get(cat, cat)) for cat in CATEGORY_CODE_MAP],
				value=DEFAULT_CATEGORIES,
			)
			submit_btn = gr.Button("処理開始", variant="primary")

		# 右側のカラム（結果表示部分）
		with gr.Column(scale=2):
			with gr.Tabs():
				with gr.Tab("結果表示"):
					with gr.Row():
						with gr.Column():
							gr.Markdown("### 原文とマスキング結果の比較")
							original_display = gr.HTML(label="原文", elem_classes="text-display")
							masked_display = gr.HTML(
								label="マスキング済みテキスト", elem_classes="text-display"
							)

					with gr.Row():
						with gr.Column():
							gr.Markdown("### GPT要約と復号結果の比較")
							gpt_display = gr.HTML(
								label="GPT要約（マスキング済み）", elem_classes="text-display"
							)
							decoded_display = gr.HTML(label="復号後のテキスト", elem_classes="text-display")

	# カスタムCSS
	css = """
    .text-display {
        background-color: #2b2b2b;
        color: #ffffff;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        font-size: 14px;
        line-height: 1.6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .text-display span {
        color: #000000;
        font-weight: 500;
    }
    
    .gradio-container {
        background-color: #1e1e1e;
        color: #ffffff;
    }

    .tabs {
        margin-top: 20px;
        background-color: #2b2b2b;
        border-radius: 8px;
        padding: 10px;
    }
    
    .markdown-text {
        color: #ffffff !important;
    }
    
    button.primary {
        background-color: #0d6efd;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        color: white;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    
    button.primary:hover {
        background-color: #0b5ed7;
    }

    /* タブのスタイル */
    .tab-nav {
        background-color: #2b2b2b !important;
        border-radius: 8px 8px 0 0;
    }
    
    .tab-nav button {
        color: #ffffff !important;
    }
    
    /* 入力テキストエリアのスタイル */
    textarea {
        background-color: #2b2b2b !important;
        color: #ffffff !important;
        border: 1px solid #444 !important;
    }
    
    /* チェックボックスグループのスタイル */
    .checkbox-group {
        background-color: #2b2b2b !important;
        color: #ffffff !important;
        padding: 10px;
        border-radius: 8px;
    }
    
    label {
        color: #ffffff !important;
    }
    """

	gr.HTML(f"<style>{css}</style>")

	def run_process(text: str, selected_categories: list[str]) -> tuple:
		result = process_text(text, selected_categories)
		if "error" in result:
			error_message = (
				'<div class="text-display" '
				'style="background-color: #ffdddd; color: #d8000c;">'
				f'{result["error"]}</div>'
			)
			return error_message, "", "", ""

		return (
			f'<div class="text-display">{result["original"]}</div>',
			f'<div class="text-display">{result["masked"]}</div>',
			f'<div class="text-display">{result["gpt_response"]}</div>',
			f'<div class="text-display">{result["decoded"]}</div>',
		)

	submit_btn.click(
		fn=run_process,
		inputs=[input_text, categories],
		outputs=[original_display, masked_display, gpt_display, decoded_display],
	)

if __name__ == "__main__":
	demo.launch()
