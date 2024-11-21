# gui.py

import atexit
import os
import re
import tempfile

import gradio as gr
import pandas as pd
import requests
from dotenv import load_dotenv

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

# スタイル定義
STYLE_DEFINITIONS = """
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
	font-family: 'Noto Sans', sans-serif;
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

button.secondary {
	background-color: #6c757d;
	border: none;
	padding: 10px 20px;
	border-radius: 5px;
	color: white;
	font-weight: bold;
	transition: background-color 0.3s;
}

button.secondary:hover {
	background-color: #5a6268;
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

/* オプションタブのスタイル */
.option-section {
	background-color: #2b2b2b !important;
	color: #ffffff !important;
	padding: 20px;
	border-radius: 8px;
	margin-bottom: 20px;
}

.option-section .textbox {
	background-color: #3c3c3c !important;
	color: #ffffff !important;
	border: 1px solid #555 !important;
}

.option-section .json-display {
	background-color: #3c3c3c !important;
	color: #ffffff !important;
	border: 1px solid #555 !important;
	border-radius: 8px;
	padding: 10px;
}

.error-message {
	background-color: #ffdddd !important;
	color: #d8000c !important;
	border: 1px solid #ffb2b2 !important;
}

.success-message {
	background-color: #dff0d8 !important;
	color: #3c763d !important;
	border: 1px solid #d6e9c6 !important;
}

/* データフレームのスタイル */
.dataframe {
	background-color: #2b2b2b !important;
	color: #ffffff !important;
}

.dataframe th {
	background-color: #1e1e1e !important;
	color: #ffffff !important;
}

.dataframe td {
	background-color: #2b2b2b !important;
	color: #ffffff !important;
}

/* JSONビューアのスタイル */
.json-viewer {
	background-color: #2b2b2b !important;
	color: #ffffff !important;
	border: 1px solid #444 !important;
	border-radius: 8px;
	padding: 10px;
}

/* スクロールバーのスタイル */
::-webkit-scrollbar {
	width: 10px;
	height: 10px;
}

::-webkit-scrollbar-track {
	background: #1e1e1e;
}

::-webkit-scrollbar-thumb {
	background: #888;
	border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
	background: #555;
}
"""


def create_error_display(error_msg: str) -> str:
	"""エラーメッセージの表示形式を統一"""
	return f"""
		<div class="text-display error-message" 
			style="background-color: #ffdddd; color: #d8000c;">
			<strong>エラー:</strong> {error_msg}
		</div>
	"""


def create_success_display(text: str) -> str:
	"""成功メッセージの表示形式を統一"""
	return f'<div class="text-display success-message">{text}</div>'


def mask_text(
	text: str, categories: list[str] = None, key_values: dict = None, values: list = None
) -> dict:
	"""テキストをマスキング処理する関数"""
	try:
		response = requests.post(
			MASKING_ENDPOINT,
			headers={"Content-Type": "application/json"},
			json={
				"text": text,
				"categories_to_mask": categories or [],
				"mask_style": "descriptive",
				"key_values_to_mask": key_values or {},
				"values_to_mask": values or [],
			},
		)
		response.raise_for_status()
		return response.json()
	except requests.exceptions.RequestException as e:
		print("Masking error:", str(e))
		return {"error": f"マスキングエラー: {str(e)}"}


def decode_text(masking_response: dict) -> str:
	"""マスキングされたテキストをデコードする関数"""
	try:
		#  リクエストの構造を確認
		# print(
		#     "Decoding request:",
		#     json.dumps(masking_response, ensure_ascii=False, indent=2),
		# )

		# エンティティマッピングの形式を整える
		formatted_mapping = {}
		for mask_token, info in masking_response["entity_mapping"].items():
			formatted_mapping[mask_token] = {
				"original_text": info.get("original_text", ""),
				"masked_text": mask_token,
				"category": info.get("category", ""),
				"source": info.get("source", ""),
			}

		# デコードリクエストを作成
		decode_request_dict = {
			"masked_text": masking_response["masked_text"],
			"entity_mapping": formatted_mapping,
		}

		# print(
		# 	"Formatted decode request:",
		# 	json.dumps(decode_request_dict, ensure_ascii=False, indent=2),
		# )

		# デコードリクエストを送信
		response = requests.post(
			DECODE_ENDPOINT,
			headers={"Content-Type": "application/json"},
			json=decode_request_dict,
		)
		response.raise_for_status()
		# print("Response Status:", response.status_code)
		# print("Response Content:", response.text)

		return response.json()["decoded_text"]

	except requests.exceptions.RequestException as e:
		print("Decoding error:", str(e))
		return create_error_display(f"デコードエラー: {str(e)}")
	except Exception as e:
		print("Unexpected error:", str(e))
		return create_error_display(f"予期せぬエラー: {str(e)}")


def gpt_ask(masked_text: str) -> str:
	"""GPTにテキストを送信して応答を受け取る関数"""
	try:
		gpt = GPTHandler(OPENAI_API_KEY)
		messages = [
			{"role": "system", "content": "あなたは要約や分析を行うアシスタントです。"},
			{
				"role": "user",
				"content": f"以下のテキストを3行で要約してください：\n\n{masked_text}",
			},
		]
		return gpt.ask(messages)
	except Exception as e:
		print("GPT error:", str(e))
		return f"GPTエラー: {str(e)}"


def highlight_differences(
	original_text: str, masking_result: dict, highlight_color: str = None
) -> tuple:
	"""マスキング結果を元に、変更された部分をハイライトする関数"""
	if "entity_mapping" not in masking_result:
		return original_text, masking_result.get("masked_text", original_text)

	# 元のテキストとマスキングされたテキストを準備
	masked_text = masking_result["masked_text"]

	# HTML タグを除去した元のテキストを作成
	clean_original = re.sub(r"<[^>]+>", "", original_text)

	# ハイライト位置を保存するリスト
	highlights_original = []
	highlights_masked = []

	# エンティティを長さの降順でソート
	entities = sorted(
		[(token, info) for token, info in masking_result["entity_mapping"].items()],
		key=lambda x: len(x[1].get("original_text", x[1].get("text", ""))),
		reverse=True,
	)

	# 元のテキストでのハイライト済みの位置を管理するリスト
	occupied_positions_original = []

	# 各エンティティの位置を特定
	for _mask_token, info in entities:
		original_txt = info.get("original_text", info.get("text", ""))
		category = info.get("category", "")
		color = CATEGORY_COLOR_MAP.get(category, highlight_color or "yellow")

		# 元のテキストでの位置を特定 (クリーンなテキストを使用)
		for match in re.finditer(re.escape(original_txt), clean_original):
			start_pos = match.start()
			end_pos = match.end()
			# 重複を避ける
			overlap = False
			for occupied_start, occupied_end in occupied_positions_original:
				if not (end_pos <= occupied_start or start_pos >= occupied_end):
					overlap = True
					break
			if not overlap:
				highlights_original.append((start_pos, end_pos, original_txt, color))
				occupied_positions_original.append((start_pos, end_pos))

	# マスクされたテキストでのハイライト済みの位置を管理するリスト
	occupied_positions_masked = []

	# マスクされたテキストでの位置を特定
	for mask_token, info in entities:
		category = info.get("category", "")
		color = CATEGORY_COLOR_MAP.get(category, highlight_color or "yellow")
		for match in re.finditer(re.escape(mask_token), masked_text):
			start_pos = match.start()
			end_pos = match.end()
			# 重複を避ける
			overlap = False
			for occupied_start, occupied_end in occupied_positions_masked:
				if not (end_pos <= occupied_start or start_pos >= occupied_end):
					overlap = True
					break
			if not overlap:
				highlights_masked.append((start_pos, end_pos, mask_token, color))
				occupied_positions_masked.append((start_pos, end_pos))

	# 位置でソート（後ろから処理）
	highlights_original.sort(reverse=True)
	highlights_masked.sort(reverse=True)

	# ハイライトを適用 (クリーンなテキストから開始)
	result_original = clean_original
	for start, end, text, color in highlights_original:
		result_original = (
			result_original[:start]
			+ (
				f'<span style="background-color: {color}; padding: 2px 4px; '
				f'border-radius: 4px;">{text}</span>'
			)
			+ result_original[end:]
		)

	result_masked = masked_text
	for start, end, text, color in highlights_masked:
		result_masked = (
			result_masked[:start]
			+ (
				f'<span style="background-color: {color}; padding: 2px 4px; '
				f'border-radius: 4px;">{text}</span>'
			)
			+ result_masked[end:]
		)

	return result_original, result_masked


def process_text(
	input_text: str,
	categories: list[str],
	key_values_to_mask: dict,
	values_to_mask: list,
) -> dict:
	"""テキスト処理の全体プロセスを実行する関数"""
	if not input_text.strip():
		return {"error": "入力テキストが空です。"}

	try:
		# マスキング処理
		masking_result = mask_text(input_text, categories, key_values_to_mask, values_to_mask)
		# print("Masking Result:", json.dumps(masking_result, ensure_ascii=False, indent=2))
		print("masking_result:", masking_result)
		if "error" in masking_result:
			return {"error": masking_result["error"]}

		# GPT要約
		gpt_response = gpt_ask(masking_result["masked_text"])
		# print("GPT Response:", gpt_response)

		# GPT応答用のマッピング作成とデコード処理
		gpt_result_mapping = {
			"masked_text": gpt_response,
			"entity_mapping": masking_result["entity_mapping"],
		}

		decoded_response = decode_text(gpt_result_mapping)
		# print("Decoded Response:", decoded_response)

		# エラー処理
		if (
			isinstance(decoded_response, str)
			and 'class="text-display error-message"' in decoded_response
		):
			return {"error": decoded_response}

		# ハイライト処理
		highlighted_original, highlighted_masked = highlight_differences(
			input_text, masking_result
		)

		highlighted_decoded, highlighted_gpt = highlight_differences(
			decoded_response,
			{"masked_text": gpt_response, "entity_mapping": masking_result["entity_mapping"]},
		)

		return {
			"original": highlighted_original,
			"masked": highlighted_masked,
			"gpt_response": highlighted_gpt,
			"decoded": highlighted_decoded,
			"entity_mapping": masking_result["entity_mapping"],
		}

	except Exception as e:
		print("Process error:", str(e))
		return {"error": create_error_display(f"処理エラー: {str(e)}")}


def re_decode(entity_mapping_df, masked_text):
	"""エンティティマッピングを使用して再デコードを行う関数"""
	try:
		if not isinstance(masked_text, str):
			# Gradioコンポーネントから値を取得
			masked_text = masked_text.value

		# マスクされたテキストからHTML要素を除去
		clean_masked_text = re.sub(r"<[^>]+>", "", masked_text)

		# DataFrameをエンティティマッピングに変換
		entity_mapping = {}
		for _, row in entity_mapping_df.iterrows():
			mask_token = row["マスクトークン"]
			entity_mapping[mask_token] = {
				"original_text": row["元のテキスト"],
				"masked_text": mask_token,
				"category": row["カテゴリ"],
				"source": row["ソース"],
			}

		# デコードリクエストを作成
		decode_request = {"masked_text": clean_masked_text, "entity_mapping": entity_mapping}

		# デコード処理
		decoded_text = decode_text(decode_request)
		# print("Re-decoded text:", decoded_text)

		# エラーチェック
		if (
			isinstance(decoded_text, str)
			and 'class="text-display error-message"' in decoded_text
		):
			return decoded_text

		# 結果をハイライト表示
		highlighted_decoded, _ = highlight_differences(
			decoded_text, {"masked_text": clean_masked_text, "entity_mapping": entity_mapping}
		)
		return create_success_display(highlighted_decoded)

	except Exception as e:
		print("Re-decode error:", str(e))
		return create_error_display(f"再デコードエラー: {str(e)}")


def convert_entity_df_to_mapping(df: pd.DataFrame) -> dict:
	"""DataFrameをエンティティマッピングに変換"""
	mapping = {}
	for _, row in df.iterrows():
		mapping[row["マスクトークン"]] = {
			"original_text": row["元のテキスト"],
			"category": row["カテゴリ"],
			"source": row["ソース"],
		}
	return mapping


def convert_mapping_to_entity_df(mapping: dict) -> pd.DataFrame:
	"""エンティティマッピングをDataFrameに変換"""
	records = []
	for mask_token, info in mapping.items():
		records.append(
			{
				"マスクトークン": mask_token,
				"元のテキスト": info.get("original_text", ""),
				"カテゴリ": info.get("category", ""),
				"ソース": info.get("source", ""),
			}
		)
	return pd.DataFrame(records)


# Gradio インターフェースの作成
with gr.Blocks(
	theme=gr.themes.Soft(
		primary_hue="blue",
		secondary_hue="gray",
	)
) as demo:
	# 状態管理用の変数
	state = gr.State(
		{"key_values_to_mask": {}, "values_to_mask": [], "last_masking_result": None}
	)

	gr.Markdown(
		"""
		# テキストマスキング & 要約システム

		テキストの匿名化と要約を行うシステムです。
		"""
	)

	with gr.Tabs():
		# メイン処理タブ
		with gr.Tab("テキスト処理"):
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
						choices=list(CATEGORY_CODE_MAP.keys()),
						value=[
							key for key, code in CATEGORY_CODE_MAP.items() if code in DEFAULT_CATEGORIES
						],
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
									decoded_display = gr.HTML(
										label="復号後のテキスト", elem_classes="text-display"
									)

							with gr.Row():
								with gr.Column():
									original_download = gr.File(label="原文をダウンロード", interactive=False)
									masked_download = gr.File(
										label="マスク済みテキストをダウンロード", interactive=False
									)
								with gr.Column():
									gpt_download = gr.File(label="GPT要約をダウンロード", interactive=False)
									decoded_download = gr.File(
										label="復号後のテキストをダウンロード", interactive=False
									)

							with gr.Row():
								with gr.Column():
									gr.Markdown("### エンティティマッピング")
									entity_display = gr.Dataframe(
										headers=["マスクトークン", "元のテキスト", "カテゴリ", "ソース"],
										datatype=["str", "str", "str", "str"],
										interactive=True,
										label="エンティティマッピングを編集",
									)

							# 再デコードボタンを追加
							with gr.Row():
								with gr.Column():
									re_decode_btn = gr.Button("エンティティを再デコード")
								with gr.Column():
									re_decoded_display = gr.HTML(
										label="再デコード後のテキスト", elem_classes="text-display"
									)

		# オプションタブ
		with gr.Tab("オプション"):
			gr.Markdown("### キー・バリューのマスキング設定")
			with gr.Row():
				key_input = gr.Textbox(label="マスクするキー", placeholder="例：株式会社Lightblue")
				value_input = gr.Textbox(label="置換後の値", placeholder="例：lead tech")
			with gr.Row():
				add_key_value_btn = gr.Button("追加/更新", variant="primary")
				delete_key_value_btn = gr.Button("削除", variant="secondary")

			key_values_display = gr.JSON(label="現在のキー・バリュー設定", value={})

			gr.Markdown("### 値のマスキング設定（UUID置換）")
			with gr.Row():
				value_to_mask_input = gr.Textbox(
					label="マスクする値", placeholder="例：RAG Ready診断"
				)
			with gr.Row():
				add_value_btn = gr.Button("追加", variant="primary")
				delete_value_btn = gr.Button("削除", variant="secondary")

			values_display = gr.JSON(label="現在の値設定", value=[])

	# 一時ファイルのパスを保存するリスト
	temporary_files = []

	def create_file(content: str, filename: str) -> str:
		"""内容を持つ一時ファイルを作成し、そのパスを返す"""
		temp = tempfile.NamedTemporaryFile(
			delete=False, suffix=f"_{filename}.txt", mode="w", encoding="utf-8"
		)
		temp.write(content)
		temp.close()
		temporary_files.append(temp.name)
		return temp.name

	def cleanup_temp_files():
		for file_path in temporary_files:
			try:
				os.remove(file_path)
			except Exception as e:
				print(f"Failed to delete temporary file {file_path}: {e}")

	# アプリケーション終了時に一時ファイルをクリーンアップ
	atexit.register(cleanup_temp_files)

	def run_process(text: str, selected_categories: list[str], state) -> tuple:
		selected_codes = [CATEGORY_CODE_MAP.get(cat, cat) for cat in selected_categories]
		result = process_text(
			text, selected_codes, state["key_values_to_mask"], state["values_to_mask"]
		)

		if "error" in result:
			return (
				create_error_display(result["error"]),
				"",
				"",
				"",
				None,
				None,
				None,
				None,  # File outputsをNoneに設定
				None,
				state,
			)

		# エンティティマッピングをDataFrame形式に変換
		entity_df = convert_mapping_to_entity_df(result["entity_mapping"])

		# テキストファイルを生成
		original_file = create_file(result["original"], "original")
		masked_file = create_file(result["masked"], "masked")
		gpt_file = create_file(result["gpt_response"], "gpt_response")
		decoded_file = create_file(result["decoded"], "decoded")

		# 状態を更新
		state["last_masking_result"] = result

		return (
			result["original"],
			result["masked"],
			result["gpt_response"],
			result["decoded"],
			original_file,
			masked_file,
			gpt_file,
			decoded_file,
			entity_df,
			state,
		)

	# オプション関連の関数
	def update_key_values(key: str, value: str, state) -> tuple:
		"""キー・バリュー設定を更新"""
		if key and value:
			state["key_values_to_mask"][key] = value
		return gr.update(value=state["key_values_to_mask"]), state

	def delete_key_value(key: str, state) -> tuple:
		"""キー・バリューを削除"""
		if key in state["key_values_to_mask"]:
			del state["key_values_to_mask"][key]
		return gr.update(value=state["key_values_to_mask"]), state

	def update_values_to_mask(value: str, state) -> tuple:
		"""UUIDマスク対象の値を追加"""
		if value and value not in state["values_to_mask"]:
			state["values_to_mask"].append(value)
		return gr.update(value=state["values_to_mask"]), state

	def delete_value_to_mask(value: str, state) -> tuple:
		"""UUIDマスク対象の値を削除"""
		if value in state["values_to_mask"]:
			state["values_to_mask"].remove(value)
		return gr.update(value=state["values_to_mask"]), state

	# イベントハンドラの接続
	submit_btn.click(
		fn=run_process,
		inputs=[input_text, categories, state],
		outputs=[
			original_display,
			masked_display,
			gpt_display,
			decoded_display,
			original_download,
			masked_download,
			gpt_download,
			decoded_download,
			entity_display,
			state,
		],
	)

	add_key_value_btn.click(
		fn=update_key_values,
		inputs=[key_input, value_input, state],
		outputs=[key_values_display, state],
	)

	delete_key_value_btn.click(
		fn=delete_key_value,
		inputs=[key_input, state],
		outputs=[key_values_display, state],
	)

	add_value_btn.click(
		fn=update_values_to_mask,
		inputs=[value_to_mask_input, state],
		outputs=[values_display, state],
	)

	delete_value_btn.click(
		fn=delete_value_to_mask,
		inputs=[value_to_mask_input, state],
		outputs=[values_display, state],
	)

	re_decode_btn.click(
		fn=re_decode,
		inputs=[entity_display, masked_display],
		outputs=[re_decoded_display],
	)

	# スタイルシートの適用
	gr.HTML("<style>" + STYLE_DEFINITIONS + "</style>")

if __name__ == "__main__":
	demo.launch()
