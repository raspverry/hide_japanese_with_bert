from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy
import hashlib
from typing import List, Optional, Dict
import os

# GPU 사용 비활성화 (필요한 경우)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# FastAPI アプリケーションの初期化
app = FastAPI(title="固有表現マスキングAPI")

# カテゴリのマッピング定義
CATEGORY_MAPPING = {
    "ORG": ["Company", "Corporation_Other"],
    "PERSON": ["Person"],
    "LOCATION": ["Province"],
    "PRODUCT": ["Product_Other"],
    "POSITION": ["Position_Vocation"]
}

# spaCy モデルのロード
try:
    nlp = spacy.load('ja_ginza_bert_large')
except OSError:
    raise HTTPException(
        status_code=500,
        detail="日本語モデルのロードに失敗しました。ja_ginza_bert_largeがインストールされているか確認してください。"
    )

# リクエストモデルの定義
class MaskingRequest(BaseModel):
    text: str
    categories_to_mask: Optional[List[str]] = None
    
# レスポンスモデルの定義
class MaskingResponse(BaseModel):
    encoded_text: str
    entity_dictionary: Dict[str, Dict[str, str]]
    decoded_text: str
    debug_info: Dict[str, List[Dict[str, str]]]

def get_ginza_categories(categories: List[str]) -> List[str]:
    """標準カテゴリをGiNZAのカテゴリに変換"""
    if categories is None:
        return None
        
    ginza_categories = []
    for category in categories:
        if category in CATEGORY_MAPPING:
            ginza_categories.extend(CATEGORY_MAPPING[category])
    return ginza_categories

def encode_named_entities(text: str, categories_to_mask: Optional[List[str]] = None):
    """指定されたカテゴリの固有表現をハッシュキーに変換します"""
    doc = nlp(text)
    ent_dict = {}
    encoded_text = text
    
    # GiNZAのカテゴリに変換
    ginza_categories = get_ginza_categories(categories_to_mask)
    
    # デバッグ情報収集用
    detected_entities = []
    
    # 検出された固有表現とそのカテゴリを出力
    for ent in doc.ents:
        detected_entities.append({
            "text": ent.text,
            "label": ent.label_
        })
        print(f"Found entity: {ent.text} (Category: {ent.label_})")
        
        # カテゴリが指定されていない場合、または指定されたカテゴリに含まれる場合
        if ginza_categories is None or ent.label_ in ginza_categories:
            hash_key = hashlib.sha256(ent.text.encode()).hexdigest()[:10]
            ent_dict[hash_key] = {
                "text": ent.text,
                "label": ent.label_
            }
            encoded_text = encoded_text.replace(ent.text, f'{{{hash_key}}}')

    return encoded_text, ent_dict, detected_entities

def decode_named_entities(encoded_text: str, ent_dict: Dict[str, Dict[str, str]]):
    """ハッシュキーを元の固有表現に復元します"""
    decoded_text = encoded_text
    for hash_key, entity_info in ent_dict.items():
        decoded_text = decoded_text.replace(f'{{{hash_key}}}', entity_info["text"])
    return decoded_text

@app.post("/mask_entities", response_model=MaskingResponse)
async def mask_entities(request: MaskingRequest):
    """
    テキスト内の固有表現をマスキングするエンドポイント
    """
    try:
        print(f"Requested categories to mask: {request.categories_to_mask}")
        
        # エンコーディングの実行
        encoded_text, ent_dict, detected_entities = encode_named_entities(
            request.text, 
            request.categories_to_mask
        )
        
        # デコーディングの実行（検証用）
        decoded_text = decode_named_entities(encoded_text, ent_dict)
        
        return MaskingResponse(
            encoded_text=encoded_text,
            entity_dictionary=ent_dict,
            decoded_text=decoded_text,
            debug_info={"detected_entities": detected_entities}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"テキスト処理中にエラーが発生しました: {str(e)}"
        )

if __name__ == "__main__":
    # テスト用コード
    test_text = """最先端アルゴリズムの社会実装に取り組むAIスタートアップ、株式会社Lightblue(代表取締役:園田亜斗夢、本社:東京都千代田区、以下「Lightblue」)は、生成AIの導入効果を最大化するための診断サービス「RAG Ready診断」をリリースいたしました。"""
    doc = nlp(test_text)
    print("=== Available Entity Categories ===")
    for ent in doc.ents:
        print(f"Entity: {ent.text} -> Category: {ent.label_}")
    print("================================")
    print("Available standard categories:", list(CATEGORY_MAPPING.keys()))
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)