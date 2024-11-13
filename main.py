import spacy
import hashlib

print(spacy.prefer_gpu())



# 日本語のNLPモデルをロード
nlp = spacy.load('ja_ginza_bert_large')

# エンコード関数：固有名詞をハッシュキーに変換し、カテゴリ情報を追加
def encode_named_entities(text):
    doc = nlp(text)
    ent_dict = {}
    encoded_text = text

    for ent in doc.ents:
        hash_key = hashlib.sha256(ent.text.encode()).hexdigest()[:10]  # ハッシュキーを生成
        ent_dict[hash_key] = {
            "text": ent.text,
            "label": ent.label_  # カテゴリ情報を追加
        }
        encoded_text = encoded_text.replace(ent.text, f'{{{hash_key}}}')  # テキスト内の固有名詞をハッシュキーに置き換え

    return encoded_text, ent_dict

# デコード関数：ハッシュキーを元の固有名詞に復元
def decode_named_entities(encoded_text, ent_dict):
    decoded_text = encoded_text
    for hash_key, entity_info in ent_dict.items():
        decoded_text = decoded_text.replace(f'{{{hash_key}}}', entity_info["text"])  # ハッシュキーを元のテキストに置き換え
    return decoded_text

# テスト用の文章
text = """最先端アルゴリズムの社会実装に取り組むAIスタートアップ、株式会社Lightblue(代表取締役:園田亜斗夢、本社:東京都千代田区、以下「Lightblue」)は、生成AIの導入効果を最大化するための診断サービス「RAG Ready診断」をリリースいたしました。
本診断は、生成AIとRAG(Retrieval-Augmented Generation)の導入準備が整っているかを評価し、企業の生成AI活用の次なるステップをサポートすることを目的としています。"""

encoded_text, ent_dict = encode_named_entities(text)
print("Original Text:", text)
print("Encoded Text:", encoded_text)  # エンコードされたテキストを表示
print("Entity Dictionary:", ent_dict)  # ハッシュキー、固有名詞、カテゴリを含む辞書を表示

# デコードのテスト
decoded_text = decode_named_entities(encoded_text, ent_dict)
print("Decoded Text:", decoded_text)  # デコードされたテキストを表示

