# 固有表現マスキングAPI

テキスト内の固有表現（組織名、人名など）を自動的に検出し、ハッシュ化されたキーでマスキングするRESTful APIです。マスキングされた固有表現は、後で元のテキストに復元することができます。

## 必要要件

- Python 3.10
- CUDA 11.7
- FastAPI
- GiNZA BERT Large model

## インストール方法

1. Python 3.10とCUDA 11.7が正しくインストールされていることを確認してください。

2. 必要なパッケージをインストールします：

```bash
# GiNZA BERT Largeモデルのインストール
uv pip install "https://github.com/megagonlabs/ginza/releases/download/v5.2.0/ja_ginza_bert_large-5.2.0b1-py3-none-any.whl"

# SpaCyのCUDAサポートをインストール
uv pip install -U spacy[cuda117]
```

## 使用方法

1. APIサーバーを起動します：

```bash
uvicorn main:app --reload
```

2. エンドポイントにリクエストを送信します：

```bash
curl -X POST "http://localhost:8000/mask_entities" \
-H "Content-Type: application/json" \
-d '{
    "text": "マスキングしたいテキスト",
    "categories_to_mask": ["ORG", "PERSON"]
}'
```

## 参考リンク

- [GiNZA BERT Large使用ガイド](https://megagonlabs.github.io/ginza/)
- [CUDA環境でのSpaCyセットアップ](https://qiita.com/CaughC/items/a67a2c8e3bad9c81833)
- [固有表現抽出の実装例](https://zenn.dev/ncdc/articles/824c6c9bbbf93ac)

## APIの機能

### エンドポイント: `/mask_entities`

#### リクエストボディ
```json
{
    "text": "マスキング対象のテキスト",
    "categories_to_mask": ["ORG", "PERSON"]  // オプション
}
```

#### レスポンス
```json
{
    "encoded_text": "マスキングされたテキスト",
    "entity_dictionary": {
        "ハッシュキー": {
            "text": "元のテキスト",
            "label": "カテゴリ"
        }
    },
    "decoded_text": "復元されたテキスト"
}
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細については[LICENSE](LICENSE)ファイルを参照してください。

