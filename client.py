# client.py
import requests
from typing import Optional, List, Dict
from app.utils import convert_masking_response_to_decode_request
from gpt_handler import GPTHandler
from dotenv import load_dotenv
import os

load_dotenv()

def mask_text(text: str, categories: Optional[List[str]] = None) -> dict:
    """テキストをマスキング処理"""
    response = requests.post(
        "http://localhost:8000/mask_text",
        headers={"Content-Type": "application/json"},
        json={
            "text": text,
            "categories_to_mask": categories or [],
            "mask_style": "descriptive"
        }
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Masking Error: {response.status_code} - {response.text}")

def decode_text(masking_result: Dict) -> str:
    """マスキングされたテキストをデコード"""
    # マスキング結果からデコードリクエストを生成
    decode_request = convert_masking_response_to_decode_request(masking_result)
    
    response = requests.post(
        "http://localhost:8000/decode_text",
        headers={"Content-Type": "application/json"},
        json={
            "masked_text": decode_request.masked_text,
            "entity_mapping": decode_request.entity_mapping
        }
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
        print("original Text:")
        print(text)
        
        # 1. テキストのマスキング
        masking_result = mask_text(
            text=text,
            categories=["ORG", "PERSON", "LOCATION", "POSITION"]
        )
        
        print("Masked Text:")
        print(masking_result["masked_text"])
        
        # 2. GPTに質問
        key=os.getenv("OPENAI_API_KEY")
        gpt = GPTHandler(key)
        messages = [
            {"role": "system", "content": "あなたは要約や分析を行うアシスタントです。"},
            {"role": "user", "content": f"以下のテキストを3行で要約してください：\n\n{masking_result['masked_text']}"}
        ]
        gpt_response = gpt.ask(messages)
        
        print("\nGPT Response (Masked):")
        print(gpt_response)
        
        # 3. GPTの応答をデコード
        # マスキング結果を新しい構造に変換
        gpt_masking_result = {
            "masked_text": gpt_response,
            "entity_mapping": masking_result["entity_mapping"]
        }
        decoded_response = decode_text(gpt_masking_result)
        
        print("\nDecoded Response:")
        print(decoded_response)
        
    except Exception as e:
        print(f"Error: {str(e)}")