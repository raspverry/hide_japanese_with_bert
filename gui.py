# gui.py

import atexit
import os
import re
import tempfile
import time

import gradio as gr
import pandas as pd
import requests
from dotenv import load_dotenv

from gpt_handler import GPTHandler


# Load environment variables
load_dotenv()

# Get OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
	raise ValueError("The environment variable OPENAI_API_KEY is not set.")

# Masking service endpoints
MASKING_ENDPOINT = "http://localhost:8000/mask_text"
DECODE_ENDPOINT = "http://localhost:8000/decode_text"

# Default and available categories
DEFAULT_CATEGORIES: list[str] = []

# Mapping for category display in Japanese
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
	"国名": "COUNTRY",
}

DEFAULT_COLOR = "rgba(255, 255, 0, 0.6)"  # Default highlight color

# Color mapping by category
CATEGORY_COLOR_MAP = {
	"ORG": "rgba(255, 105, 180, 0.6)",  # Organization: Hot Pink
	"PERSON": "rgba(255, 165, 0, 0.6)",  # Person: Orange
	"LOCATION": "rgba(50, 205, 50, 0.6)",  # Location: Lime Green
	"POSITION": "rgba(30, 144, 255, 0.6)",  # Position: Dodger Blue
	"DATE": "rgba(147, 112, 219, 0.6)",  # Date: Purple
	"EVENT": "rgba(255, 215, 0, 0.6)",  # Event: Gold
	"PRODUCT": "rgba(220, 20, 60, 0.6)",  # Product: Crimson
	"NORP": "rgba(70, 130, 180, 0.6)",  # NORP: Steel Blue
	"FACILITY": "rgba(34, 139, 34, 0.6)",  # Facility: Forest Green
	"GPE": "rgba(244, 164, 96, 0.6)",  # GPE: Sandy Brown
	"LAW": "rgba(186, 85, 211, 0.6)",  # Law: Medium Purple
	"LANGUAGE": "rgba(255, 140, 0, 0.6)",  # Language: Dark Orange
	"MONEY": "rgba(46, 139, 87, 0.6)",  # Money: Sea Green
	"PERCENT": "rgba(65, 105, 225, 0.6)",  # Percent: Royal Blue
	"TIME": "rgba(138, 43, 226, 0.6)",  # Time: Blue Violet
	"QUANTITY": "rgba(100, 149, 237, 0.6)",  # Quantity: Cornflower Blue
	"ORDINAL": "rgba(219, 112, 147, 0.6)",  # Ordinal: Pale Violet Red
	"CARDINAL": "rgba(218, 165, 32, 0.6)",  # Cardinal: Goldenrod
	"EMAIL": "rgba(255, 20, 147, 0.6)",  # Email: Deep Pink
	"PHONE": "rgba(95, 158, 160, 0.6)",  # Phone: Cadet Blue
	"PROJECT": "rgba(255, 127, 80, 0.6)",  # Project: Coral
	"DEPARTMENT": "rgba(199, 21, 133, 0.6)",  # Department: Medium Violet Red
	"COUNTRY": "rgba(0, 128, 0, 0.6)",  # Country: Green
}

# Define styles with CSS variables
STYLE_DEFINITIONS = """
:root {
	--background-color: #1e1e1e;
	--text-color: #ffffff;
	--background-secondary: #2b2b2b;
	--border-color: #444;
	--input-background: #2b2b2b;
	--input-text-color: #ffffff;
	--highlight-color: #ffffff;
}

:root[data-theme="dark"] {
	--background-color: #1e1e1e;
	--text-color: #ffffff;
	--background-secondary: #2b2b2b;
	--border-color: #444;
	--input-background: #2b2b2b;
	--input-text-color: #ffffff;
	--highlight-color: #ffffff;
}

:root[data-theme="light"] {
	--background-color: #ffffff;
	--text-color: #000000;
	--background-secondary: #d4d4d4;
	--border-color: ##a1a1a1;
	--input-background: #ffffff;
	--input-text-color: #000000;
	--highlight-color: #000000;
}

.gradio-container {
	background-color: var(--background-color);
	color: var(--text-color);
	font-family: 'Noto Sans', sans-serif;
}

.text-display {
	background-color: var(--background-secondary);
	color: var(--text-color);
	border: 1px solid var(--border-color);
	border-radius: 8px;
	padding: 15px;
	margin: 10px 0;
	font-size: 14px;
	line-height: 1.6;
	box-shadow: 0 2px 4px rgba(0,0,0,0.2);
	white-space: pre-wrap;
}

.text-display span {
	color: var(--highlight-color);
	font-weight: 500;
}

.result-container {
	background-color: #f3f4f6;
	padding: 1rem;
	border-radius: 0.5rem;
}

textarea {
	background-color: var(--input-background) !important;
	color: var(--input-text-color) !important;
	border: 1px solid var(--border-color) !important;
}

.char-counter {
	font-size: 14px;
	color: var(--text-color);
	margin-left: 10px;
	white-space: nowrap;
}

label {
	color: var(--text-color) !important;
}

input[type="checkbox"] + label {
	color: var(--text-color) !important;
}

button.primary {
	background-color: #0d6efd;
	border: none;
	padding: 8px 16px;
	border-radius: 5px;
	color: white;
	font-weight: bold;
	transition: background-color 0.3s;
	font-size: 14px;
}

button.primary:hover {
	background-color: #0b5ed7;
}

button.secondary {
	background-color: #6c757d;
	border: none;
	padding: 8px 16px;
	border-radius: 5px;
	color: white;
	font-weight: bold;
	transition: background-color 0.3s;
	font-size: 14px;
}

button.secondary:hover {
	background-color: #5a6268;
}

/* Scrollbar Styles */
::-webkit-scrollbar {
	width: 10px;
	height: 10px;
}

::-webkit-scrollbar-track {
	background: var(--background-color);
}

::-webkit-scrollbar-thumb {
	background: #888;
	border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
	background: #555;
}

/* Copy Button Styles */
.copy-button {
	background: none;
	border: none;
	cursor: pointer;
	color: var(--text-color);
	font-size: 16px;
	padding: 2px;
	width: 30px;
	height: 30px;
	display: flex;
	align-items: center;
	justify-content: center;
	min-width: 30px !important;
}

.copy-button:hover {
	color: #dddddd;
}

.relative-container {
	position: relative;
}
"""

# JavaScript for copying and theme toggling
copy_and_theme_js = """
(() =>  {
	const themeCheckbox = document.getElementById('theme_switch')
	if (!themeCheckbox) {
		console.error('Theme switch checkbox not found!')
		return
	}

	function updateTheme(isDark) {
		const theme = isDark ? 'dark' : 'light'

		// Set theme at root level
		document.documentElement.dataset.theme = theme

		// Update body and container
		document.body.classList.remove('dark', 'light')
		document.body.classList.add(theme)

		const gradioContainer = document.querySelector('.gradio-container')
		if (gradioContainer) {
			gradioContainer.classList.remove('dark', 'light')
			gradioContainer.classList.add(theme)
		}

		const theme_switch = document.querySelector('#theme_switch')
		console.log(theme_switch.checked)

		const input_textbox = document.querySelector('#input_textbox')
		const categories_checkbox = document.querySelector('#categories_checkbox')

		const original_download = document.querySelector('#original_download')
		const masked_download = document.querySelector('#masked_download')
		const gpt_download = document.querySelector('#gpt_download')
		const decoded_download = document.querySelector('#decoded_download')
		
		const original_display = document.querySelector('#original_display')
		const masked_display = document.querySelector('#masked_display')
		const gpt_display = document.querySelector('#gpt_display')
		const decoded_display = document.querySelector('#decoded_display')

		if(theme == 'dark'){
			theme_switch.style.backgroundColor = '#1e2936'
			input_textbox.style.backgroundColor = '#1e2936'
			categories_checkbox.style.backgroundColor = '#1e2936'

			original_download.style.backgroundColor = '#1e2936'
			masked_download.style.backgroundColor = '#1e2936'
			gpt_download.style.backgroundColor = '#1e2936'
			decoded_download.style.backgroundColor = '#1e2936'
   
			original_display.style.backgroundColor = '#1e2936'
			masked_display.style.backgroundColor = '#1e2936'
			gpt_display.style.backgroundColor = '#1e2936'
			decoded_display.style.backgroundColor = '#1e2936'

			theme_switch.checked = true
		}
		else{
			theme_switch.style.backgroundColor = '#e5e7eb'
			input_textbox.style.backgroundColor = '#e5e7eb'
			categories_checkbox.style.backgroundColor = '#e5e7eb'

			original_download.style.backgroundColor = '#e5e7eb'
			masked_download.style.backgroundColor = '#e5e7eb'
			gpt_download.style.backgroundColor = '#e5e7eb'
			decoded_download.style.backgroundColor = '#e5e7eb'

			original_display.style.backgroundColor = '#e5e7eb'
			masked_display.style.backgroundColor = '#e5e7eb'
			gpt_display.style.backgroundColor = '#e5e7eb'
			decoded_display.style.backgroundColor = '#e5e7eb'

			theme_switch.checked = false
		}

		// Update elements
		const elementsToUpdate = document.querySelectorAll(
			'.gradio-container, textarea, input, select, .gr-box, ' +
			'.gr-panel, .gr-form, .gr-input, .text-display, ' +
			'.gr-check-radio, table, th, td, .markdown, ' +
			'.contain, button, label, .tabs'
		)

		/*
		elementsToUpdate.forEach(el => {
			if (el.classList.contains('gradio-container')) {
				el.style.backgroundColor = getComputedStyle(document.documentElement)
					.getPropertyValue('--background-color')
			} else {
				el.style.backgroundColor = getComputedStyle(document.documentElement)
					.getPropertyValue('--background-secondary')
			}
			el.style.color = getComputedStyle(document.documentElement)
				.getPropertyValue('--text-color')

			if (['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName)) {
				el.style.backgroundColor = getComputedStyle(document.documentElement)
					.getPropertyValue('--input-background')
				el.style.color = getComputedStyle(document.documentElement)
					.getPropertyValue('--input-text-color')
			}
		})
		*/

		localStorage.setItem('theme', theme)

	}

	// Initialize theme
	const savedTheme = localStorage.getItem('theme') || 'light'
	themeCheckbox.checked = savedTheme === 'dark'
	updateTheme(themeCheckbox.checked)

	// Theme change event
	themeCheckbox.addEventListener('change', e => {
		updateTheme(e.target.checked)
	})


	// Copy to clipboard function
	function CopyToClipboard(text) {
		var textArea = document.createElement("textarea");
		textArea.value = text;
		document.body.appendChild(textArea);
		textArea.select();
		document.execCommand('copy');
		document.body.removeChild(textArea);
	}

	// copy event
	const copyOriginal = document.getElementById('copy_original')
	copyOriginal.addEventListener('click', e => {
		const textElement = document.getElementById('original_display')
		if (!textElement) return
		console.log(textElement.textContent)
		CopyToClipboard(textElement.textContent.trim())
	})
	const copyMasked = document.getElementById('copy_masked')
	copyMasked.addEventListener('click', e => {
		const textElement = document.getElementById('masked_display')
		if (!textElement) return
		console.log(textElement.textContent)
		CopyToClipboard(textElement.textContent.trim())
	})

	const copyGpt = document.getElementById('copy_gpt')
	copyGpt.addEventListener('click', e => {
		const textElement = document.getElementById('gpt_display')
		if (!textElement) return
		console.log(textElement.textContent)
		CopyToClipboard(textElement.textContent.trim())
	})

	const copyDecoded = document.getElementById('copy_decoded')
	copyDecoded.addEventListener('click', e => {
		const textElement = document.getElementById('decoded_display')
		if (!textElement) return
		console.log(textElement.textContent)
		CopyToClipboard(textElement.textContent.trim())
	})

})
"""


def create_error_display(error_msg: str) -> str:
	"""Standardize the error message display format"""
	return f"""
		<div class="text-display error-message">
			<strong>Error:</strong> {error_msg}
		</div>
	"""


def create_success_display(text: str) -> str:
	"""Standardize the success message display format"""
	return f'<div class="text-display success-message">{text}</div>'


def mask_text(
	text: str,
	categories: list[str] | None = None,
	key_values: dict | None = None,
	values: list | None = None,
) -> dict:
	"""Function to perform masking on the text"""
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
		return {"error": f"Masking error: {str(e)}"}


def decode_text(masking_response: dict) -> str:
	"""Function to decode masked text"""
	try:
		# Create decode request
		decode_request_dict = {
			"masked_text": masking_response["masked_text"],
			"entity_mapping": masking_response["entity_mapping"],
		}

		# Send decode request
		response = requests.post(
			DECODE_ENDPOINT,
			headers={"Content-Type": "application/json"},
			json=decode_request_dict,
		)
		response.raise_for_status()

		return response.json()["decoded_text"]

	except requests.exceptions.RequestException as e:
		print("Decoding error:", str(e))
		return create_error_display(f"Decoding error: {str(e)}")
	except Exception as e:
		print("Unexpected error:", str(e))
		return create_error_display(f"Unexpected error: {str(e)}")


def gpt_ask(masked_text: str) -> str:
	"""Function to send text to GPT and receive a response"""
	try:
		gpt = GPTHandler(OPENAI_API_KEY)
		messages = [
			{
				"role": "system",
				"content": "You are an assistant who performs summaries and analysis.",
			},
			{
				"role": "user",
				"content": f"""Please summarize the following text in 3 lines:
							\n\n{masked_text}
							""",
			},
		]
		return gpt.ask(messages)
	except Exception as e:
		print("GPT error:", str(e))
		return f"GPT error: {str(e)}"


def highlight_differences(
	original_text: str, masking_result: dict, highlight_color: str = None
) -> tuple:
	"""Function to highlight changes based on masking results"""
	if "entity_mapping" not in masking_result:
		return original_text, masking_result.get("masked_text", original_text)

	# Prepare original and masked texts
	masked_text = masking_result["masked_text"]

	# Create clean original text without HTML tags
	clean_original = re.sub(r"<[^>]+>", "", original_text)

	# Lists to save highlight positions
	highlights_original = []
	highlights_masked = []

	# Sort entities in descending order of length
	entities = sorted(
		[(token, info) for token, info in masking_result["entity_mapping"].items()],
		key=lambda x: len(x[1].get("original_text", x[1].get("text", ""))),
		reverse=True,
	)

	# Manage occupied positions in the original text
	occupied_positions_original = []

	# Identify positions in the original text
	for _mask_token, info in entities:
		original_txt = info.get("original_text", info.get("text", ""))
		category = info.get("category", "")
		color = CATEGORY_COLOR_MAP.get(category, highlight_color or DEFAULT_COLOR)

		# Find positions in the clean original text
		for match in re.finditer(re.escape(original_txt), clean_original):
			start_pos = match.start()
			end_pos = match.end()
			# Avoid overlaps
			overlap = False
			for occupied_start, occupied_end in occupied_positions_original:
				if not (end_pos <= occupied_start or start_pos >= occupied_end):
					overlap = True
					break
			if not overlap:
				highlights_original.append((start_pos, end_pos, original_txt, color))
				occupied_positions_original.append((start_pos, end_pos))

	# Manage occupied positions in the masked text
	occupied_positions_masked = []

	# Identify positions in the masked text
	for mask_token, info in entities:
		category = info.get("category", "")
		color = CATEGORY_COLOR_MAP.get(category, highlight_color or DEFAULT_COLOR)
		for match in re.finditer(re.escape(mask_token), masked_text):
			start_pos = match.start()
			end_pos = match.end()
			# Avoid overlaps
			overlap = False
			for occupied_start, occupied_end in occupied_positions_masked:
				if not (end_pos <= occupied_start or start_pos >= occupied_end):
					overlap = True
					break
			if not overlap:
				highlights_masked.append((start_pos, end_pos, mask_token, color))
				occupied_positions_masked.append((start_pos, end_pos))

	# Sort positions (process from the end)
	highlights_original.sort(reverse=True)
	highlights_masked.sort(reverse=True)

	# Apply highlights starting from clean text
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
	"""Function to execute the entire text processing"""
	if not input_text.strip():
		return {"error": "The input text is empty."}

	try:
		# Masking
		masking_result = mask_text(
			input_text, categories, key_values_to_mask, values_to_mask
		)

		if "error" in masking_result:
			return {"error": masking_result["error"]}

		# GPT summarization
		gpt_response = gpt_ask(masking_result["masked_text"])

		# Create mapping for GPT response and decode
		gpt_result_mapping = {
			"masked_text": gpt_response,
			"entity_mapping": masking_result["entity_mapping"],
		}

		decoded_response = decode_text(gpt_result_mapping)

		# Error handling
		if (
			isinstance(decoded_response, str)
			and 'class="text-display error-message"' in decoded_response
		):
			return {"error": decoded_response}

		# Highlighting
		highlighted_original, highlighted_masked = highlight_differences(
			input_text, masking_result
		)

		highlighted_decoded, highlighted_gpt = highlight_differences(
			decoded_response,
			{
				"masked_text": gpt_response,
				"entity_mapping": masking_result["entity_mapping"],
			},
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
		return {"error": create_error_display(f"Processing error: {str(e)}")}


def re_decode(entity_mapping_df, masked_text):
	"""Function to perform re-decoding using the entity mapping"""
	try:
		if not masked_text.strip():
			raise ValueError("The masked text is empty.")

		if not isinstance(masked_text, str):
			# Get value from Gradio component
			masked_text = masked_text.value

		# Remove HTML elements from masked text
		clean_masked_text = re.sub(r"<[^>]+>", "", masked_text)

		# Convert DataFrame to entity mapping
		entity_mapping = {}
		for _, row in entity_mapping_df.iterrows():
			mask_token = row["マスクトークン"]
			entity_mapping[mask_token] = {
				"original_text": row["元のテキスト"],
				"masked_text": mask_token,
				"category": row["カテゴリ"],
				"source": row["ソース"],
			}

		# Create decode request
		decode_request = {
			"masked_text": clean_masked_text,
			"entity_mapping": entity_mapping,
		}

		# Perform decode
		decoded_text = decode_text(decode_request)

		# Error check
		if (
			isinstance(decoded_text, str)
			and 'class="text-display error-message"' in decoded_text
		):
			return decoded_text

		# Display result with highlights
		highlighted_decoded, _ = highlight_differences(
			decoded_text,
			{"masked_text": clean_masked_text, "entity_mapping": entity_mapping},
		)
		return create_success_display(highlighted_decoded)

	except Exception as e:
		print("Re-decode error:", str(e))
		return create_error_display(f"Re-decode error: {str(e)}")


def convert_entity_df_to_mapping(df: pd.DataFrame) -> dict:
	"""Convert DataFrame to entity mapping"""
	mapping = {}
	for _, row in df.iterrows():
		mapping[row["マスクトークン"]] = {
			"original_text": row["元のテキスト"],
			"category": row["カテゴリ"],
			"source": row["ソース"],
		}
	return mapping


def convert_mapping_to_entity_df(mapping: dict) -> pd.DataFrame:
	"""Convert entity mapping to DataFrame"""
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


def delete_entity_and_redecode(
	entity_df: pd.DataFrame, selected_index: int, masked_text: str
) -> tuple:
	"""エンティティを削除して再デコードを行う"""
	try:
		if selected_index is None:  # 行が選択されていない場合
			return entity_df, create_error_display("行を選択してください。")

		selected_index = int(selected_index)

		if selected_index < 0 or selected_index >= len(entity_df):
			return entity_df, create_error_display("有効な行を選択してください。")

		# 指定された行を削除
		entity_df = entity_df.drop(index=selected_index).reset_index(drop=True)

		# DataFrameをマッピングに変換
		entity_mapping = {}
		for _, row in entity_df.iterrows():
			mask_token = row["マスクトークン"]
			entity_mapping[mask_token] = {
				"original_text": row["元のテキスト"],
				"masked_text": mask_token,
				"category": row["カテゴリ"],
				"source": row["ソース"],
			}

		# マスクされたテキストからHTMLタグを削除
		clean_masked_text = re.sub(r"<[^>]+>", "", masked_text)

		# デコード実行
		decode_request = {
			"masked_text": clean_masked_text,
			"entity_mapping": entity_mapping,
		}

		# デコード結果を取得
		decoded_text = decode_text(decode_request)

		# ハイライト処理
		highlighted_decoded, _ = highlight_differences(
			decoded_text,
			{"masked_text": clean_masked_text, "entity_mapping": entity_mapping},
		)

		return entity_df, create_success_display(highlighted_decoded)

	except Exception as e:
		print("Delete and redecode error:", str(e))
		return entity_df, create_error_display(f"削除・再デコードエラー: {str(e)}")


def toggle_theme(checkbox_value: bool, state: dict) -> tuple:
	"""テーマのtoggle"""
	theme = "dark" if checkbox_value else "light"
	state["theme"] = theme

	return state, gr.HTML()


# Create Gradio interface
with gr.Blocks(
	theme=gr.themes.Soft(
		primary_hue="blue",
		secondary_hue="gray",
	),
	js=copy_and_theme_js,
	css=STYLE_DEFINITIONS,
) as demo:
	# State management variable
	state = gr.State(
		{
			"key_values_to_mask": {},
			"values_to_mask": [],
			"last_masking_result": None,
			"theme": "dark",
		}
	)

	# State for selected row index in entity mapping
	selected_row_index = gr.State()

	gr.Markdown(
		"""
		# テキストマスキング & 要約システム

		テキストの匿名化と要約を行うシステムです。
		"""
	)

	# Theme toggle switch
	with gr.Row():
		theme_switch = gr.Checkbox(
			label="ダークモード",
			value=True,
			elem_id="theme_switch",
		)

		theme_switch.change(
			fn=toggle_theme,
			inputs=[theme_switch, state],
			outputs=[state, gr.HTML()],  # gr._js.Js()の代わりに gr.HTML()利用
		)

	with gr.Tabs():
		# Main processing tab
		with gr.Tab("テキスト処理"):
			with gr.Row():
				# Left column (input)
				with gr.Column(scale=1):
					input_text = gr.Textbox(
						label="入力テキスト",
						placeholder="ここに日本語テキストを入力してください...",
						lines=10,
						elem_id="input_textbox",
						max_length=5000,  # 最大文字数を設定
					)
					# 文字数カウンター
					char_counter = gr.HTML(
						value="(0 / 5000)",
						elem_id="char_counter",
					)
					categories = gr.CheckboxGroup(
						label="マスキングカテゴリ",
						choices=list(CATEGORY_CODE_MAP.keys()),
						value=[
							key
							for key, code in CATEGORY_CODE_MAP.items()
							if code in DEFAULT_CATEGORIES
						],
						elem_id="categories_checkbox",
					)
					submit_btn = gr.Button("処理開始", variant="primary")

				# Right column (results)
				with gr.Column(scale=2):
					with gr.Tabs():
						with gr.Tab("結果表示"):
							processing_time_display = gr.HTML()
							with gr.Row():
								with gr.Column():
									gr.Markdown("### 原文とマスキング結果の比較")
									# Original display and copy button
									with gr.Row():
										with gr.Column(scale=10):
											original_display = gr.HTML(
												label="原文",
												elem_classes="text-display",
												elem_id="original_display",
											)
										with gr.Column(scale=1, min_width=30):
											copy_original_btn = gr.Button(
												"📋",
												variant="secondary",
												elem_id="copy_original",
												elem_classes="copy-button",
											)

									# Masked text display and copy button
									with gr.Row():
										with gr.Column(scale=10):
											masked_display = gr.HTML(
												label="マスキング済みテキスト",
												elem_classes="text-display",
												elem_id="masked_display",
											)
										with gr.Column(scale=1, min_width=30):
											copy_masked_btn = gr.Button(
												"📋",
												variant="secondary",
												elem_id="copy_masked",
												elem_classes="copy-button",
											)

							with gr.Row():
								with gr.Column():
									gr.Markdown("### GPT要約と復号結果の比較")
									# GPT summary display and copy button
									with gr.Row():
										with gr.Column(scale=10):
											gpt_display = gr.HTML(
												label="GPT要約（マスキング済み）",
												elem_classes="text-display",
												elem_id="gpt_display",
											)
										with gr.Column(scale=1, min_width=30):
											copy_gpt_btn = gr.Button(
												"📋",
												variant="secondary",
												elem_id="copy_gpt",
												elem_classes="copy-button",
											)

									# Decoded text display and copy button
									with gr.Row():
										with gr.Column(scale=10):
											decoded_display = gr.HTML(
												label="復号後のテキスト",
												elem_classes="text-display",
												elem_id="decoded_display",
											)
										with gr.Column(scale=1, min_width=30):
											copy_decoded_btn = gr.Button(
												"📋",
												variant="secondary",
												elem_id="copy_decoded",
												elem_classes="copy-button",
											)

							with gr.Row():
								with gr.Column():
									original_download = gr.File(
										label="原文をダウンロード",
										interactive=False,
										elem_id="original_download",
									)
									masked_download = gr.File(
										label="マスク済みテキストをダウンロード",
										interactive=False,
										elem_id="masked_download",
									)
								with gr.Column():
									gpt_download = gr.File(
										label="GPT要約をダウンロード",
										interactive=False,
										elem_id="gpt_download",
									)
									decoded_download = gr.File(
										label="復号後のテキストをダウンロード",
										interactive=False,
										elem_id="decoded_download",
									)

							with gr.Row():
								with gr.Column():
									gr.Markdown("### エンティティマッピング")
									entity_display = gr.Dataframe(
										headers=[
											"マスクトークン",
											"元のテキスト",
											"カテゴリ",
											"ソース",
										],
										datatype=["str", "str", "str", "str"],
										interactive=True,
										label="エンティティマッピングを編集",
										elem_id="entity_display",
									)

									def update_selected_row_index(evt: gr.SelectData):
										index = evt.index

										if index is None:
											return None
										elif isinstance(index, list) or isinstance(
											index, tuple
										):
											row_index = index[0]
											return row_index
										else:
											return index

									entity_display.select(
										fn=update_selected_row_index,
										inputs=None,
										outputs=selected_row_index,
									)

									delete_entity_btn = gr.Button(
										"選択したエンティティを削除",
										variant="secondary",
									)

							# Add re-decode button
							with gr.Row():
								with gr.Column():
									re_decode_btn = gr.Button(
										"エンティティを再デコード"
									)
								re_process_btn = gr.Button("再処理", variant="primary")
								with gr.Column():
									re_decoded_display = gr.HTML(
										label="再デコード後のテキスト",
										elem_classes="text-display",
										elem_id="re_decoded_display",
									)

		# Options tab
		with gr.Tab("オプション"):
			gr.Markdown("### キー・バリューのマスキング設定")
			with gr.Row():
				key_input = gr.Textbox(
					label="マスクするキー",
					placeholder="例：株式会社Lightblue",
					elem_id="key_input",
				)
				value_input = gr.Textbox(
					label="置換後の値",
					placeholder="例：lead tech",
					elem_id="value_input",
				)
			with gr.Row():
				add_key_value_btn = gr.Button(
					"追加/更新", variant="primary", elem_id="add_key_value_btn"
				)
				delete_key_value_btn = gr.Button(
					"削除", variant="secondary", elem_id="delete_key_value_btn"
				)

			key_values_display = gr.JSON(
				label="現在のキー・バリュー設定", value={}, elem_id="key_values_display"
			)

			gr.Markdown("### 値のマスキング設定（UUID置換）")
			with gr.Row():
				value_to_mask_input = gr.Textbox(
					label="マスクする値",
					placeholder="例：RAG Ready診断",
					elem_id="value_to_mask_input",
				)
			with gr.Row():
				add_value_btn = gr.Button(
					"追加", variant="primary", elem_id="add_value_btn"
				)
				delete_value_btn = gr.Button(
					"削除", variant="secondary", elem_id="delete_value_btn"
				)

			values_display = gr.JSON(
				label="現在の値設定", value=[], elem_id="values_display"
			)

	# List to store paths of temporary files
	temporary_files = []

	def create_file(content: str, file_type: str) -> str:
		"""Create a temporary file with content and return its path"""
		from datetime import datetime

		timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
		filename = f"{timestamp}_{file_type}.txt"
		temp = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8")
		temp.write(content)
		temp.close()

		# 실제 파일명으로 변경
		final_path = os.path.join(os.path.dirname(temp.name), filename)

		os.rename(temp.name, final_path)
		temporary_files.append(final_path)
		return final_path

	def cleanup_temp_files():
		for file_path in temporary_files:
			try:
				os.remove(file_path)
			except Exception as e:
				print(f"Failed to delete temporary file {file_path}: {e}")

	# Clean up temporary files when the application exits
	atexit.register(cleanup_temp_files)

	def run_process(text: str, selected_categories: list[str], state) -> tuple:
		start_time = time.time()

		selected_codes = [
			CATEGORY_CODE_MAP.get(cat, cat) for cat in selected_categories
		]
		result = process_text(
			text, selected_codes, state["key_values_to_mask"], state["values_to_mask"]
		)

		# 処理時間を計算
		processing_time = time.time() - start_time

		processing_time_html = gr.HTML(
			f"""<div style="padding: 10px; text-align: right;">
				処理時間: {processing_time:.2f}秒
			</div>"""
		)

		if "error" in result:
			return (
				result["error"],
				"",
				"",
				"",
				None,
				None,
				None,
				None,  # File outputs set to None
				None,
				state,
			)

		# Convert entity mapping to DataFrame
		entity_df = convert_mapping_to_entity_df(result["entity_mapping"])

		# Generate text files
		original_file = create_file(result["original"], "original")
		masked_file = create_file(result["masked"], "masked")
		gpt_file = create_file(result["gpt_response"], "gpt_response")
		decoded_file = create_file(result["decoded"], "decoded")

		# Update state
		state["last_masking_result"] = result

		return (
			processing_time_html,
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

	def re_process(original_text: str, entity_mapping_df: pd.DataFrame, state) -> tuple:
		"""再処理を実行"""
		try:
			print("original_text:", original_text)
			if original_text is None or not original_text.strip():
				raise ValueError("Original text is missing.")

			# DataFrameをマッピングに変換
			entity_mapping = {}
			for _, row in entity_mapping_df.iterrows():
				mask_token = row["マスクトークン"]
				entity_mapping[mask_token] = {
					"original_text": row["元のテキスト"],
					"masked_text": mask_token,
					"category": row["カテゴリ"],
					"source": row["ソース"],
				}

			# マスクされたテキストを取得
			masked_text = apply_entity_mapping(original_text, entity_mapping)

			# GPTにマスキングされたテキストを送信して応答を受け取る
			gpt_response = gpt_ask(masked_text)

			# GPT応答とデコード用のマッピングを作成
			gpt_result_mapping = {
				"masked_text": gpt_response,
				"entity_mapping": entity_mapping,
			}
			decoded_response = decode_text(gpt_result_mapping)

			# ハイライト処理
			highlighted_original, highlighted_masked = highlight_differences(
				original_text,
				{"masked_text": masked_text, "entity_mapping": entity_mapping},
			)
			highlighted_decoded, highlighted_gpt = highlight_differences(
				decoded_response,
				{
					"masked_text": gpt_response,
					"entity_mapping": entity_mapping,
				},
			)

			# 結果を返す
			return (
				highlighted_original,  # original_display
				highlighted_masked,  # masked_display
				highlighted_gpt,  # gpt_display
				highlighted_decoded,  # decoded_display
				state,
			)

		except Exception as e:
			print("Re-process error:", str(e))
			return (create_error_display(f"再処理エラー: {str(e)}"), "", "", "", state)

	def apply_entity_mapping(text: str, entity_mapping: dict) -> str:
		"""エンティティマッピングを適用してテキストをマスキング"""
		# エンティティを長い順にソート
		entities = sorted(
			entity_mapping.items(),
			key=lambda x: len(x[1].get("original_text", "")),
			reverse=True,
		)

		masked_text = text
		for mask_token, info in entities:
			original_text = info.get("original_text", "")
			if original_text:
				# マスクトークンを元のテキストに置換
				masked_text = re.sub(re.escape(original_text), mask_token, masked_text)
		return masked_text

	# Functions related to options
	def update_key_values(key: str, value: str, state) -> tuple:
		"""Update key-value settings"""
		if key and value:
			state["key_values_to_mask"][key] = value
		return gr.update(value=state["key_values_to_mask"]), state

	def delete_key_value(key: str, state) -> tuple:
		"""Delete a key-value"""
		if key in state["key_values_to_mask"]:
			del state["key_values_to_mask"][key]
		return gr.update(value=state["key_values_to_mask"]), state

	def update_values_to_mask(value: str, state) -> tuple:
		"""Add a value to be masked with UUID"""
		if value and value not in state["values_to_mask"]:
			state["values_to_mask"].append(value)
		return gr.update(value=state["values_to_mask"]), state

	def delete_value_to_mask(value: str, state) -> tuple:
		"""Delete a value to be masked with UUID"""
		if value in state["values_to_mask"]:
			state["values_to_mask"].remove(value)
		return gr.update(value=state["values_to_mask"]), state

	# 文字数カウンターを更新する関数
	def update_char_counter(text: str) -> str:
		length = len(text)
		return f"({length} / 5000)"

	# Connect event handlers
	submit_btn.click(
		fn=run_process,
		inputs=[input_text, categories, state],
		outputs=[
			processing_time_display,
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

	# 文字数カウンターの更新を接続
	input_text.change(
		fn=update_char_counter,
		inputs=input_text,
		outputs=char_counter,
	)

	delete_entity_btn.click(
		fn=delete_entity_and_redecode,
		inputs=[entity_display, selected_row_index, masked_display],
		outputs=[entity_display, re_decoded_display],
	)

	re_process_btn.click(
		fn=re_process,
		inputs=[input_text, entity_display, state],
		outputs=[original_display, masked_display, gpt_display, decoded_display, state],
	)


if __name__ == "__main__":
	demo.launch()
