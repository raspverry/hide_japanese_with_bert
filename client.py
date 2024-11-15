# client.py

import requests
import json
from typing import Optional, List
from app.utils import convert_masking_response_to_decode_request

def mask_text(text: str, categories: Optional[List[str]] = None, mask_style: str = "descriptive") -> dict:
    """テキストをマスキング処理する"""
    url = "http://localhost:8000/mask_text"
    
    payload = {
        "text": text,
        "categories_to_mask": categories or [],
        "mask_style": mask_style
    }
    
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return {}

def decode_text(masking_result: dict) -> str:
    """マスキングされたテキストを元に戻す"""
    url = "http://localhost:8000/decode_text"
    
    # MaskingResponseをDecodeRequestに変換
    decode_request = convert_masking_response_to_decode_request(masking_result)
    
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={
            "masked_text": decode_request.masked_text,
            "entity_mapping": decode_request.entity_mapping
        }
    )
    
    if response.status_code == 200:
        return response.json()["decoded_text"]
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return ""

def print_entities(masking_result: dict):
    """マスキング結果のエンティティ情報を表示"""
    print("\nDetected Entities:")
    print("-" * 60)
    for entity in masking_result["debug_info"]["detected_entities"]:
        print(f"Text: {entity['original']}")
        print(f"Category: {entity['category']}")
        print(f"Source: {entity['source']}")
        print(f"Token: {entity['mask_token']}")
        print("-" * 60)

if __name__ == "__main__":
    # マスキングしたいテキスト
    text = """
    最先端アルゴリズムの社会実装に取り組むAIスタートアップ、株式会社Lightblue(代表取締役:園田亜斗夢、本社:東京都千代田区、以下「Lightblue」)は、
    生成AIの導入効果を最大化するための診断サービス「RAG Ready診断」をリリースいたしました。
    """
    text= """2023年の夏、私は東京大学で開催されたAI国際会議に出席し、その後京都に移動して金閣寺を訪れました。会議にはGoogleの山田太郎氏や、京都大学の佐藤花子教授など、多くの著名な研究者が参加していました。"""
    
    # マスキング処理
    result = mask_text(
        text=text,
        categories=[],  # マスキングしたいカテゴリ
        mask_style="descriptive"  # "descriptive" または "simple"
    )
    print("original text:")
    print(text)
    # マスキング結果を表示
    print("\nMasked Text:")
    print(result["masked_text"])
    
    # エンティティ情報を表示
    print_entities(result)
    
    # デコード処理
    decoded = decode_text(result)
    print("\nDecoded Text:")
    print(decoded)