import os
import re
import json
import logging
from typing import List, Optional, Dict, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
from contextlib import asynccontextmanager

import spacy
from spacy.tokens import Doc
from spacy.language import Language
from spacy.pipeline import EntityRuler

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import uvicorn

# PyTorch 関連の警告を無視
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ロギング設定
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedMaskingRequest(BaseModel):
    """マスキングリクエストのモデル"""
    text: str = Field(..., max_length=5000, description="マスキング対象のテキスト")
    categories_to_mask: Optional[List[str]] = Field(
        None,
        description="マスキングするカテゴリのリスト"
    )
    mask_style: Optional[str] = Field(
        "descriptive",
        description='"descriptive" または "simple" のマスキングスタイル'
    )


class Position(BaseModel):
    """位置情報のモデル"""
    start: int
    end: int


class DetectedEntity(BaseModel):
    """検出されたエンティティのモデル"""
    original: str
    category: str
    mask_token: str
    position: Position
    source: str


class DebugInfo(BaseModel):
    """デバッグ情報のモデル"""
    detected_entities: List[DetectedEntity]


class MaskingResponse(BaseModel):
    """マスキングレスポンスのモデル"""
    masked_text: str
    entity_mapping: Dict[str, Dict[str, str]]
    debug_info: DebugInfo
    
class DecodeRequest(BaseModel):
    """デコードリクエストのモデル"""
    masked_text: str = Field(..., description="マスキングされたテキスト")
    entity_mapping: Dict[str, Dict[str, str]] = Field(..., description="マスキングトークンと元のテキストのマッピング")


class DecodeResponse(BaseModel):
    """デコードレスポンスのモデル"""
    decoded_text: str
    

@dataclass
class Entity:
    """エンティティ情報を保持するデータクラス"""
    text: str
    category: str
    start: int
    end: int
    priority: int = 0
    source: str = "rule"  # "rule" または "ginza"


class RuleBasedMasker:
    """ルールベースのマスキング処理を行うクラス"""

    def __init__(self, rules_file: str):
        """JSONファイルからルールを読み込む"""
        if not os.path.exists(rules_file):
            logger.error(f"ルールファイルが見つかりません: {rules_file}")
            raise FileNotFoundError(f"ルールファイルが見つかりません: {rules_file}")

        with open(rules_file, 'r', encoding='utf-8') as f:
            self.rules = json.load(f)

        # すべてのパターンを活用
        self.category_patterns = {
            "company": self.rules["rules"]["company_patterns"],
            "email": self.rules["rules"]["email_patterns"],
            "phone": self.rules["rules"]["phone_patterns"],
            "project": self.rules["rules"]["project_patterns"],
            "position": self.rules["rules"]["sensitive_terms"]["position_titles"],
            "department": self.rules["rules"]["sensitive_terms"]["departments"]
        }

        # 優先順位マップ
        self.priority_map = {
            "position": 1,    # 役職が最優先
            "company": 2,     # 次に会社名
            "person": 2,      # 人名も同じ優先度
            "department": 3,  # 部署名
            "project": 4,     # プロジェクト名
            "email": 5,       # メールアドレス
            "phone": 6,       # 電話番号
        }

        # カテゴリ正規化マップ
        self.category_map = {
            "company": "ORG",
            "email": "EMAIL",
            "phone": "PHONE",
            "project": "PROJECT",
            "position": "POSITION",
            "department": "DEPARTMENT"
        }

        # パターンをコンパイル
        self.compiled_patterns = self._compile_patterns()
        logger.info("ルールベースマスカーを初期化しました")

    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """正規表現パターンをコンパイル"""
        compiled = {}
        for category, patterns in self.category_patterns.items():
            if isinstance(patterns, list):
                compiled[category] = [
                    re.compile(p, re.UNICODE | re.IGNORECASE)
                    for p in patterns
                ]
            elif isinstance(patterns, dict):
                compiled[category] = {
                    k: [re.compile(p, re.UNICODE | re.IGNORECASE) for p in v]
                    for k, v in patterns.items()
                }
        return compiled

    def _is_excluded(self, text: str) -> bool:
        """除外パターンに該当するかチェック"""
        exclusions = self.rules['exclusions']

        # 共通単語チェック
        if text in exclusions['common_words']:
            return True

        # 安全なパターンチェック
        for pattern in exclusions['safe_patterns']:
            if re.search(pattern, text, re.UNICODE | re.IGNORECASE):
                return True

        return False

    def _find_matches(self, text: str) -> List[Entity]:
        """テキスト内のすべてのパターンマッチを検出"""
        matches = []
        processed_spans: Set[Tuple[int, int]] = set()

        # 各カテゴリのパターンでマッチング
        for category, patterns in self.compiled_patterns.items():
            if isinstance(patterns, list):
                for pattern in patterns:
                    for match in pattern.finditer(text):
                        start, end = match.span()
                        # 重複チェック
                        if not any((s <= start < e or s < end <= e)
                                   for s, e in processed_spans):
                            if not self._is_excluded(match.group()):
                                priority = self.priority_map.get(category, 99)
                                matches.append(Entity(
                                    text=match.group(),
                                    category=self.category_map.get(category, category.upper()),
                                    start=start,
                                    end=end,
                                    priority=priority,
                                    source="rule"
                                ))
                                processed_spans.add((start, end))

        return sorted(matches, key=lambda x: x.start)


class EnhancedTextMasker:
    """ルールベースと機械学習を組み合わせたマスキング処理クラス"""

    def __init__(self, rules_file: Optional[str] = None):
        """初期化"""
        rules_file = rules_file or os.getenv("MASKING_RULES_FILE", "masking_rules.json")
        self.rule_masker = RuleBasedMasker(rules_file)

        try:
            # GiNZAモデルをロード
            self.nlp = spacy.load("ja_ginza_bert_large")
            # カスタムエンティティをロード
            self._load_custom_entities()
        except Exception as e:
            logger.error(f"Spacyモデルのロードに失敗しました: {str(e)}")
            raise

        # GiNZAのカテゴリマッピング
        self.ginza_category_map = {
            "PERSON": ["Person", "PSN", "NAME", "人名"],
            "LOCATION": ["Province", "City", "GPE", "LOC", "Place", "地名"],
            "ORG": ["Company", "Corporation_Other", "Organization", "ORG"],
            "PRODUCT": ["Product_Other", "Product"],
            "DATE": ["Date", "Time_Date"],
            "TIME": ["Time"],
            "MONEY": ["Money"],
            "POSITION": ["Position_Vocation", "Position"],
            "EVENT": ["Event"]
        }

        # GiNZAの優先順位マップ
        self.ginza_priority_map = {
            "PERSON": 1,
            "ORG": 2,
            "LOCATION": 3,
            "POSITION": 1,
            "PRODUCT": 4,
            "DATE": 5,
            "TIME": 6,
            "MONEY": 7,
            "EVENT": 4
        }

        # マスキング形式
        self.mask_formats = {
            "descriptive": {
                "PERSON": "<<人物_{}>>",
                "ORG": "<<組織_{}>>",
                "LOCATION": "<<場所_{}>>",
                "PRODUCT": "<<製品_{}>>",
                "POSITION": "<<役職_{}>>",
                "DATE": "<<日付_{}>>",
                "TIME": "<<時間_{}>>",
                "MONEY": "<<金額_{}>>",
                "EMAIL": "<<メール_{}>>",
                "PHONE": "<<電話番号_{}>>",
                "PROJECT": "<<プロジェクト_{}>>",
                "DEPARTMENT": "<<部署_{}>>",
                "EVENT": "<<イベント_{}>>"
            },
            "simple": {
                "PERSON": "<<PERSON_{}>>",
                "ORG": "<<ORG_{}>>",
                "LOCATION": "<<LOC_{}>>",
                "PRODUCT": "<<PROD_{}>>",
                "POSITION": "<<POS_{}>>",
                "DATE": "<<DATE_{}>>",
                "TIME": "<<TIME_{}>>",
                "MONEY": "<<MONEY_{}>>",
                "EMAIL": "<<EMAIL_{}>>",
                "PHONE": "<<PHONE_{}>>",
                "PROJECT": "<<PROJ_{}>>",
                "DEPARTMENT": "<<DEPT_{}>>",
                "EVENT": "<<EVENT_{}>>"
            }
        }

        # 不要なテキストパターン
        self.remove_patterns = [
            (r'本社[:：]?\s*', ''),
            (r'支社[:：]?\s*', ''),
            (r'事務所[:：]?\s*', ''),
            (r',\s*本社\s*', ','),
            (r'、\s*本社\s*', '、'),
        ]

        logger.info("拡張マスカーを初期化しました")

    def _load_custom_entities(self):
        """カスタムエンティティをロード"""
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        patterns = []

        # ルールファイルからカスタムエンティティをロード
        custom_entities = self.rule_masker.rules.get("custom_entities", {})
        for label, terms in custom_entities.items():
            for term in terms:
                patterns.append({"label": label, "pattern": term})

        ruler.add_patterns(patterns)

    def _normalize_category(self, category: str) -> str:
        """カテゴリを正規化"""
        # GiNZAカテゴリの正規化
        for norm_cat, ginza_cats in self.ginza_category_map.items():
            if category in ginza_cats:
                return norm_cat

        # その他のカテゴリの正規化
        category_map = {
            "COMPANY": "ORG",
            "COMPANY_PATTERNS": "ORG",
            "SENSITIVE_TERMS_POSITION": "POSITION",
            "SENSITIVE_TERMS_DEPARTMENT": "DEPARTMENT",
        }
        return category_map.get(category, category.upper())

    def _extend_person_entity(self, text: str, start: int, end: int) -> Tuple[int, int]:
        """人名エンティティの範囲を拡張"""
        current_start = start
        current_end = end

        # 前方に拡張
        while current_end < len(text):
            char = text[current_end]
            if re.match(r'[・\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]', char):
                current_end += 1
            else:
                break

        # 後方に拡張
        while current_start > 0:
            char = text[current_start - 1]
            if re.match(r'[・\s\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]', char):
                current_start -= 1
            else:
                break

        return current_start, current_end

    def _merge_adjacent_entities(self, entities: List[Entity], text: str) -> List[Entity]:
        """隣接するエンティティを結合"""
        if not entities:
            return []

        # カテゴリごとにグループ化
        category_groups = defaultdict(list)
        for entity in sorted(entities, key=lambda x: x.start):
            category_groups[entity.category].append(entity)

        merged = []
        for category, group in category_groups.items():
            i = 0
            while i < len(group):
                current = group[i]

                # エンティティの前後のテキストを含めて分析
                start_pos = current.start
                end_pos = current.end

                # 人名の場合、前後を拡張
                if category == "PERSON":
                    start_pos, end_pos = self._extend_person_entity(text, start_pos, end_pos)

                # 隣接エンティティとの結合チェック
                while i + 1 < len(group):
                    next_entity = group[i + 1]
                    between_text = text[end_pos:next_entity.start]

                    # 結合条件チェック
                    if (len(between_text.strip()) <= 2 and
                            re.match(r'^[・\s]*$', between_text)):
                        end_pos = next_entity.end
                        i += 1
                    else:
                        break

                # 新しいエンティティを作成
                current = Entity(
                    text=text[start_pos:end_pos],
                    category=category,
                    start=start_pos,
                    end=end_pos,
                    priority=current.priority,
                    source=current.source
                )
                merged.append(current)
                i += 1

        return sorted(merged, key=lambda x: x.start)

    def _remove_overlapping_entities(self, entities: List[Entity]) -> List[Entity]:
        """重複するエンティティを削除"""
        if not entities:
            return []

        # 優先順位とカバー範囲でソート
        sorted_entities = sorted(
            entities,
            key=lambda x: (x.priority, -len(x.text), x.start)
        )

        result = []
        covered_ranges = set()

        for entity in sorted_entities:
            entity_range = set(range(entity.start, entity.end))
            overlap = entity_range & covered_ranges
            if not overlap:  # 重複なし
                result.append(entity)
                covered_ranges.update(entity_range)
            else:
                # 優先順位を比較
                overlapping_entities = [
                    e for e in result if set(range(e.start, e.end)) & entity_range
                ]
                min_priority = min(e.priority for e in overlapping_entities)
                if entity.priority < min_priority:
                    # 既存エンティティを削除
                    for e in overlapping_entities:
                        covered_ranges.difference_update(range(e.start, e.end))
                        result.remove(e)
                    result.append(entity)
                    covered_ranges.update(entity_range)

        return sorted(result, key=lambda x: x.start)

    def mask_text(self, text: str, categories: Optional[List[str]] = None,
                  mask_style: str = "descriptive") -> Tuple[str, Dict, List[Dict]]:
        """2段階マスキングを適用"""
        # テキストの前処理
        processed_text = text
        for pattern, replacement in self.remove_patterns:
            processed_text = re.sub(pattern, replacement, processed_text)

        # 1. エンティティの検出
        entities = []

        # 1.1 ルールベースのエンティティ検出
        rule_entities = self.rule_masker._find_matches(processed_text)
        entities.extend(rule_entities)

        # 1.2 GiNZAのエンティティ検出
        doc = self.nlp(processed_text)
        expanded_categories = set()
        if categories:
            for category in categories:
                if category in self.ginza_category_map:
                    expanded_categories.update(self.ginza_category_map[category])

        for ent in doc.ents:
            norm_category = self._normalize_category(ent.label_)
            if not categories or norm_category in categories:
                priority = self.ginza_priority_map.get(
                    norm_category, 99
                )
                entities.append(Entity(
                    text=ent.text,
                    category=norm_category,
                    start=ent.start_char,
                    end=ent.end_char,
                    priority=priority,
                    source="ginza"
                ))

        # 2. エンティティの統合と重複除去
        merged_entities = self._merge_adjacent_entities(entities, processed_text)
        final_entities = self._remove_overlapping_entities(merged_entities)

        # 3. マスキング適用
        entity_mapping = {}
        debug_info = []
        offset = 0
        masked_text = processed_text

        for idx, entity in enumerate(final_entities, 1):
            # マスクトークンを生成
            category = self._normalize_category(entity.category)
            mask_token = self.mask_formats[mask_style].get(category, "<<UNKNOWN_{}>>").format(idx)

            start = entity.start + offset
            end = entity.end + offset

            masked_text = masked_text[:start] + mask_token + masked_text[end:]
            offset += len(mask_token) - (end - start)

            entity_mapping[mask_token] = {
                "text": entity.text,
                "category": category,
                "source": entity.source
            }

            debug_info.append({
                "original": entity.text,
                "category": category,
                "mask_token": mask_token,
                "position": {"start": entity.start, "end": entity.end},
                "source": entity.source
            })

            logger.debug(f"マスキング適用: {entity.text} -> {mask_token} ({category})")

        return masked_text, entity_mapping, debug_info

    def decode_text(self, masked_text: str, entity_mapping: Dict[str, Dict[str, str]]) -> str:
        """マスキングされたテキストを元のテキストに復元します"""
        # マスキングトークンの長さで降順ソートして、誤置換を防ぎます
        sorted_tokens = sorted(entity_mapping.keys(), key=len, reverse=True)

        decoded_text = masked_text

        for token in sorted_tokens:
            # マスキングトークンが変形する可能性に備えて、正規表現を使用
            pattern = re.escape(token)
            decoded_text = re.sub(pattern, entity_mapping[token]['text'], decoded_text)

        return decoded_text


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        # 必要な終了処理があれば追加
        pass


# FastAPI アプリケーション設定
app = FastAPI(title="高度なテキストマスキングAPI", lifespan=lifespan)


@app.post("/mask_text", response_model=MaskingResponse)
async def mask_text_endpoint(request: EnhancedMaskingRequest):
    """テキストマスキングエンドポイント"""
    try:
        masker = EnhancedTextMasker()
        masked_text, entity_mapping, debug_info = masker.mask_text(
            request.text,
            request.categories_to_mask,
            request.mask_style
        )

        return MaskingResponse(
            masked_text=masked_text,
            entity_mapping=entity_mapping,
            debug_info=DebugInfo(detected_entities=debug_info)
        )

    except FileNotFoundError:
        logger.error("ルールファイルが見つかりません", exc_info=True)
        raise HTTPException(status_code=400, detail="ルールファイルが見つかりません。")
    except Exception as e:
        logger.error(f"エラーが発生しました: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="テキスト処理中に予期しないエラーが発生しました。"
        )


@app.post("/decode_text", response_model=DecodeResponse)
async def decode_text_endpoint(request: DecodeRequest):
    """テキストデコードエンドポイント"""
    try:
        masker = EnhancedTextMasker()
        decoded_text = masker.decode_text(request.masked_text, request.entity_mapping)
        return DecodeResponse(decoded_text=decoded_text)

    except Exception as e:
        logger.error(f"エラーが発生しました: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="デコード処理中に予期しないエラーが発生しました。"
        )



if __name__ == "__main__":
    # APIサーバー起動
    uvicorn.run(app, host="0.0.0.0", port=8000)
