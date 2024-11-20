import os
import json
import requests
import gradio as gr
from dotenv import load_dotenv
from typing import List
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
DEFAULT_CATEGORIES: List[str] = []


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

def mask_text(text: str, categories: List[str] = None) -> dict:
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
    return response.json() if response.status_code == 200 else {"error": f"マスキングエラー: {response.status_code} - {response.text}"}

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
    return response.json().get("decoded_text", "") if response.status_code == 200 else f"デコードエラー: {response.status_code} - {response.text}"

def gpt_ask(masked_text: str) -> str:
    """GPTにテキストを送信して応答を受け取る関数"""
    gpt = GPTHandler(OPENAI_API_KEY)
    messages = [
        {"role": "system", "content": "あなたは要約や分析を行うアシスタントです。"},
        {"role": "user", "content": f"以下のテキストを3行で要約してください：\n\n{masked_text}"},
    ]
    return gpt.ask(messages)


def highlight_differences(text1: str, text2: str, highlight_color: str = "yellow") -> tuple:
    """マスキングされた部分のみをハイライトする関数"""
    import re
    
    def find_mask_patterns(text):
        """マスキングパターン(<<xxx_n>>)を検出する"""
        return re.finditer(r'<<[^>]+>>', text)
    
    def find_original_text(original: str, masked: str, pattern: str, pattern_start: int, pattern_end: int) -> tuple:
        """マスキングパターンに対応する元のテキストを特定する"""
        # パターンの前後のコンテキストを取得
        context_before = masked[:pattern_start].split()[-3:] if masked[:pattern_start].split() else []
        context_after = masked[pattern_end:].split()[:3] if masked[pattern_end:].split() else []
        
        if not context_before and not context_after:
            return None, None
            
        # 前後のコンテキストを正規表現パターンに変換
        pattern_parts = []
        if context_before:
            pattern_parts.append('(?:' + '.*?'.join(map(re.escape, context_before)) + ')')
        pattern_parts.append('(.*?)')
        if context_after:
            pattern_parts.append('(?:' + '.*?'.join(map(re.escape, context_after)) + ')')
            
        search_pattern = '.*?'.join(pattern_parts)
        match = re.search(search_pattern, original)
        
        if match:
            matched_text = match.group(1)
            start_pos = match.start(1)
            end_pos = match.end(1)
            return matched_text, (start_pos, end_pos)
            
        return None, None
    
    def apply_highlights(text: str, highlights: list, color: str) -> str:
        """ハイライトを適用する"""
        result = list(text)
        # 位置がずれないように後ろから処理
        for start, end, original_text in sorted(highlights, reverse=True):
            highlight_html = f'<span style="background-color: {color};">{original_text}</span>'
            result[start:end] = highlight_html
        return ''.join(result)
    
    # マスキングパターンを検出
    original_highlights = []
    masked_patterns = []
    
    # マスクパターンとその位置を特定
    for match in find_mask_patterns(text2):
        pattern = match.group()
        pattern_start = match.start()
        pattern_end = match.end()
        
        # 元のテキストでの対応部分を特定
        original_text, pos = find_original_text(text1, text2, pattern, pattern_start, pattern_end)
        if original_text and pos:
            original_highlights.append((pos[0], pos[1], original_text))
            masked_patterns.append((pattern_start, pattern_end, pattern))
    
    # ハイライトを適用
    highlighted_original = apply_highlights(text1, original_highlights, highlight_color)
    highlighted_masked = apply_highlights(text2, masked_patterns, highlight_color)
    
    return highlighted_original, highlighted_masked


def process_text(input_text: str, categories: List[str]) -> dict:
    """テキスト処理の全体プロセスを実行する関数"""
    if not input_text.strip():
        return {"error": "入力テキストが空です。"}
        
    # マスキング処理
    masking_result = mask_text(input_text, categories)
    print(masking_result)
    if "error" in masking_result:
        return {"error": masking_result["error"]}
        
    masked_text = masking_result["masked_text"]
    entity_mapping = masking_result["entity_mapping"]
    
    # GPT要約
    gpt_response = gpt_ask(masked_text)
    
    # デコード
    decoded_response = decode_text(masking_result)
    
    # ハイライト処理
    original_highlighted, masked_highlighted = highlight_differences(input_text, masked_text, "rgba(255, 255, 0, 0.5)")
    gpt_highlighted, decoded_highlighted = highlight_differences(gpt_response, decoded_response, "rgba(144, 238, 144, 0.5)")
    
    return {
        "original": original_highlighted,
        "masked": masked_highlighted,
        "gpt_response": gpt_highlighted,
        "decoded": decoded_highlighted
    }

# Gradio インターフェースの作成
with gr.Blocks(theme=gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
)) as demo:
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
                lines=10
            )
            categories = gr.CheckboxGroup(
                label="マスキングカテゴリ",
                choices=[(cat, CATEGORY_CODE_MAP.get(cat, cat)) for cat in CATEGORY_CODE_MAP],
                value=DEFAULT_CATEGORIES
            )
            submit_btn = gr.Button("処理開始", variant="primary")
        
        # 右側のカラム（結果表示部分）
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("結果表示"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 原文とマスキング結果の比較")
                            original_display = gr.HTML(
                                label="原文",
                                elem_classes="text-display"
                            )
                            masked_display = gr.HTML(
                                label="マスキング済みテキスト",
                                elem_classes="text-display"
                            )
                        
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### GPT要約と復号結果の比較")
                            gpt_display = gr.HTML(
                                label="GPT要約（マスキング済み）",
                                elem_classes="text-display"
                            )
                            decoded_display = gr.HTML(
                                label="復号後のテキスト",
                                elem_classes="text-display"
                            )

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
    
    def run_process(text: str, selected_categories: List[str]) -> tuple:
        result = process_text(text, selected_categories)
        if "error" in result:
            error_message = f'<div class="text-display" style="background-color: #ffdddd; color: #d8000c;">{result["error"]}</div>'
            return error_message, "", "", ""
            
        return (
            f'<div class="text-display">{result["original"]}</div>',
            f'<div class="text-display">{result["masked"]}</div>',
            f'<div class="text-display">{result["gpt_response"]}</div>',
            f'<div class="text-display">{result["decoded"]}</div>'
        )

    submit_btn.click(
        fn=run_process,
        inputs=[input_text, categories],
        outputs=[original_display, masked_display, gpt_display, decoded_display]
    )

if __name__ == "__main__":
    demo.launch()
