# 固有表現マスキングAPI

テキスト内の固有表現（組織名、人名など）を自動的に検出し、ハッシュ化されたキーでマスキングするRESTful APIです。
マスキングされた固有表現は、後で元のテキストに復元することができます。

## 必要要件

- Python 3.10
- CUDA 11.7
- FastAPI
- GiNZA BERT Large model

# NVIDIAドライバーのインストール
```bash
# システムの更新
sudo apt update && sudo apt upgrade -y

# 必要なパッケージのインストール
sudo apt install -y software-properties-common wget

# NVIDIAドライバーのインストール
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot

# CUDA 11.7のインストール
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
sudo sh cuda_11.7.0_515.43.04_linux.run --toolkit --samples --silent --override

# 環境変数の設定
echo 'export PATH=/usr/local/cuda-11.7/bin:$PATH' >> ~/.zshrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH' >> ~/.zshrc
source ~/.zshrc
```

## dockerの起動
dockerなくてもインストール方法で設置できる。

dockerを使う場合、
```bash
# Dockerのインストール
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done

sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# NVIDIA Container Toolkitのインストール
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# ユーザーをdockerグループに追加
sudo usermod -aG docker $USER
newgrp docker

```

```bash
# プロジェクトコンテーナを実行
docker compose -f .devcontainer/docker-compose.yaml up -d --build

# プロジェクトコンテーナの中のzshに接続
docker exec -it hide_japanese /bin/zsh

source .venv/bin/activate

uv sync
```


## インストール方法

1. Python 3.10とCUDA 11.7が正しくインストールされていることを確認してください。

2. 必要なパッケージをインストールします：

```bash
# GiNZA BERT Largeモデルのインストール
uv pip install "https://github.com/megagonlabs/ginza/releases/download/v5.2.0/ja_ginza_bert_large-5.2.0b1-py3-none-any.whl"

# SpaCyのCUDAサポートをインストール
uv pip install -U spacy[cuda117]
```
または、以下のコマンドで依存関係を一括インストールできます：

```bash
uv sync
```

3. envファイルを生成する
```bash
cp env_dist .env
```

## 使用方法

1. APIサーバーを起動します：

```bash
uvicorn server:app --reload
```

2. エンドポイントにリクエストを送信します：

### 2.1 curlを使用

```bash
curl -X POST "http://localhost:8000/mask_text" \
-H "Content-Type: application/json" \
-d '{
    "text": "最先端アルゴリズムの社会実装に取り組むAIスタートアップ、株式会社Lightblue(代表取締役:園田亜斗夢、本社:東京都千代田区、以下「Lightblue」)は、生成AIの導入効果を最大化するための診断サービス「RAG Ready診断」をリリースいたしました。\n本診断は、生成AIとRAG(Retrieval-Augmented Generation)の導入準備が整っているかを評価し、企業の生成AI活用の次なるステップをサポートすることを目的としています。",
    "categories_to_mask": ["ORG", "PERSON"]
}'
```


for decodning

```bash
curl -X POST "http://localhost:8000/decode_text" \
-H "Content-Type: application/json" \
-d '{
    "masked_text": "最先端アルゴリズムの社会実装に取り組むAIスタートアップ、<<組織_1>>(<<役職_2>>:<<人物_3>>、<<場所_4>>、以下「<<組織_5>>」)は、生成AIの導入効果を最大化するための診断サービス「RAG Ready診断」をリリースいたしました。\n本診断は、生成AIとRAG(Retrieval-Augmented Generation)の導入準備が整っているかを評価し、企業の生成AI活用の次なるステップをサポートすることを目的としています。",
    "entity_mapping": {
        "<<組織_1>>": {"text": "株式会社Lightblue", "category": "ORG", "source": "rule"},
        "<<役職_2>>": {"text": "代表取締役", "category": "POSITION", "source": "rule"},
        "<<人物_3>>": {"text": "園田亜斗夢", "category": "PERSON", "source": "ginza"},
        "<<場所_4>>": {"text": "東京都千代田区", "category": "LOCATION", "source": "ginza"},
        "<<組織_5>>": {"text": "Lightblue", "category": "ORG", "source": "ginza"}
    }
}'
```

### 2.1 client.pyを利用

`uv run client.py`を起動したら、自動にmasking & decodingする。　　

### 2.1 cli.pyを利用

`uv run cli.py `を入力すると、'処理する日本語テキストを入力してください（終了するには Ctrl+D または Ctrl+Z を押して Enter）'
という説明が出る。そこにテキストを入力して　Ctrl+D, そしてEnterを押すと、　　
マスキング可能なカテゴリ一覧が出る。そこでマスキングしたいカテゴリーを選んで、Enterを押す。　　
自動にmasking & decodingする。　

### 2.1 GUIを利用
```bash
uv venv .gui-venv

source .gui-venv/bin/activate

#guiに必要なscriptを設置
uv add --group gui requests python-dotenv openai gradio
```


## 利用可能なマスキングカテゴリ

当APIでは、ユーザーが指定したカテゴリに基づいてテキスト内の特定の情報をマスキングすることができます。以下に、現在サポートしているマスキングカテゴリとその説明を一覧で示します。

### 1. 組織関連（Organization）
- **カテゴリ名**: `ORG`
- **説明**: 企業名、団体名、組織名などの組織に関連するエンティティをマスキングします。
- **例**: 「株式会社Lightblue」

### 2. 人物関連（Person）
- **カテゴリ名**: `PERSON`
- **説明**: 人名や役職者の名前など、人物に関連するエンティティをマスキングします。
- **例**: 「園田亜斗夢」

### 3. 電子メール（Email）
- **カテゴリ名**: `EMAIL`
- **説明**: メールアドレスをマスキングします。
- **例**: `test@example.com`

### 4. 電話番号（Phone）
- **カテゴリ名**: `PHONE`
- **説明**: 電話番号をマスキングします。
- **例**: `03-1234-5678`

### 5. プロジェクト名（Project）
- **カテゴリ名**: `PROJECT`
- **説明**: プロジェクト名やプロジェクトに関連するエンティティをマスキングします。
- **例**: 「Project-X」

### 6. 役職（Position）
- **カテゴリ名**: `POSITION`
- **説明**: 役職名や職位に関連するエンティティをマスキングします。
- **例**: 「代表取締役」

### 7. 部署（Department）
- **カテゴリ名**: `DEPARTMENT`
- **説明**: 部署名や部門名に関連するエンティティをマスキングします。
- **例**: 「開発部」

### 8. 場所（Location）
- **カテゴリ名**: `LOCATION`
- **説明**: 地名や住所に関連するエンティティをマスキングします。
- **例**: 「東京都千代田区」

### 9. 日付（Date）
- **カテゴリ名**: `DATE`
- **説明**: 日付に関連するエンティティをマスキングします。
- **例**: `2024-04-27`

### 10. 時間（Time）
- **カテゴリ名**: `TIME`
- **説明**: 時間に関連するエンティティをマスキングします。
- **例**: `午前10時`

### 11. 金額（Money）
- **カテゴリ名**: `MONEY`
- **説明**: 金額に関連するエンティティをマスキングします。
- **例**: `¥10,000`

## APIの機能

### エンドポイント: `/mask_text`

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
    "masked_text": "マスキングされたテキスト",
    "entity_mapping": {
        "<<ORG_1>>": {
            "text": "株式会社Lightblue",
            "category": "ORG",
            "source": "ginza"
        },
        "<<PERSON_2>>": {
            "text": "園田亜斗夢",
            "category": "PERSON",
            "source": "ginza"
        }
    },
    "debug_info": {
        "detected_entities": [
            {
                "original": "株式会社Lightblue",
                "category": "ORG",
                "mask_token": "<<ORG_1>>",
                "position": {
                    "start": 27,
                    "end": 42
                },
                "source": "ginza"
            },
            {
                "original": "園田亜斗夢",
                "category": "PERSON",
                "mask_token": "<<PERSON_2>>",
                "position": {
                    "start": 43,
                    "end": 48
                },
                "source": "ginza"
            }
        ]
    }
}
```

## 注意事項

- azureで作業する時は必ずSecureBootを無効化しないといけなさそうです。
- **エンドポイント名の統一**: エンドポイントは `/mask_text` です。
- **レスポンスの整合性**: レスポンス形式は、`masked_text`、`entity_mapping`、および `debug_info` を含みます。

## 参考リンク

- [GiNZA BERT Large使用ガイド](https://megagonlabs.github.io/ginza/)
- [CUDA環境でのSpaCyセットアップ](https://qiita.com/CaughC/items/a67a2c8e3bad9c81833)
- [固有表現抽出の実装例](https://zenn.dev/ncdc/articles/824c6c9bbbf93ac)
- [GiNZAまとめ](https://tt-tsukumochi.com/archives/5336)

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細については[LICENSE](LICENSE)ファイルを参照してください。

