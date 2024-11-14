# app/masking.py

import re
from typing import List, Optional, Dict, Tuple
from collections import defaultdict
import os
import json

import spacy

from app.models import Entity
from app.rules_loader import RuleBasedMasker

import structlog

# ロガーの取得
logger = structlog.get_logger(__name__)

class EnhancedTextMasker:
    """ルールベースと機械学習を組み合わせたマスキング処理クラス"""

    def __init__(self, rules_file: Optional[str] = None):
        """初期化"""
        rules_file = rules_file or "masking_rules.json"  # デフォルトのルールファイル名
        self.rule_masker = RuleBasedMasker(rules_file)

        try:
            # GiNZAモデルをロード
            self.nlp = spacy.load("ja_ginza_bert_large") 
            # カスタムエンティティをロード
            self._load_custom_entities()
        except Exception as e:
            logger.error("Spacyモデルのロードに失敗しました", error=str(e))
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
        self.mask_formats = self._load_mask_formats()

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
        logger.debug("カスタムエンティティパターンをロードしました", patterns=patterns)

    def _load_mask_formats(self) -> Dict[str, Dict[str, str]]:
        """マスキングフォーマットをロード"""
        try:
            # masking_rules.jsonの絶対パスを取得
            current_dir = os.path.dirname(os.path.abspath(__file__))
            rules_file_path = os.path.join(current_dir, '..', 'masking_rules.json')
            with open(rules_file_path, 'r', encoding='utf-8') as f:
                rules = json.load(f)
            mask_formats = rules.get("mask_formats", {})
            logger.debug("マスキングフォーマットをロードしました", mask_formats=mask_formats)
            return mask_formats
        except Exception as e:
            logger.error("マスキングフォーマットの読み込みに失敗しました", error=str(e))
            # デフォルトのフォーマットを返す
            return {
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
                    "EVENT": "<<イベント_{}>>",
                    "DOCTRINE_METHOD_OTHER": "<<方法_{}>>"
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
                    "EVENT": "<<EVENT_{}>>",
                    "DOCTRINE_METHOD_OTHER": "<<METHOD_{}>>" 
                }
            }

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
                merged_entity = Entity(
                    text=text[start_pos:end_pos],
                    category=category,
                    start=start_pos,
                    end=end_pos,
                    priority=current.priority,
                    source=current.source
                )
                merged.append(merged_entity)
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
        # ログにmask_formatsを出力
        logger.debug("マスキング処理開始", mask_style=mask_style, mask_formats=self.mask_formats)

        # テキストの前処理
        processed_text = text
        for pattern, replacement in self.remove_patterns:
            processed_text = re.sub(pattern, replacement, processed_text)
        logger.debug("テキストの前処理完了", processed_text=processed_text)

        # 1. エンティティの検出
        entities = []

        # 1.1 ルールベースのエンティティ検出
        rule_entities = self.rule_masker._find_matches(processed_text)
        entities.extend(rule_entities)
        logger.debug("ルールベース検出エンティティ", rule_entities=[e.__dict__ for e in rule_entities])

        # 1.2 GiNZAのエンティティ検出
        doc = self.nlp(processed_text)
        if categories:
            expanded_categories = set()
            for category in categories:
                if category in self.ginza_category_map:
                    expanded_categories.update(self.ginza_category_map[category])
        else:
            expanded_categories = None

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
        logger.debug("GiNZA検出エンティティ", ginza_entities=[e.__dict__ for e in entities if e.source == "ginza"])

        # 2. エンティティの統合と重複除去
        merged_entities = self._merge_adjacent_entities(entities, processed_text)
        final_entities = self._remove_overlapping_entities(merged_entities)
        logger.debug("最終エンティティ", final_entities=[e.__dict__ for e in final_entities])

        # 3. マスキング適用
        entity_mapping = {}
        debug_info = []
        offset = 0
        masked_text = processed_text

        for idx, entity in enumerate(final_entities, 1):
            # マスクトークンを生成
            category = self._normalize_category(entity.category)
            mask_token = self.mask_formats.get(mask_style, {}).get(category, "<<UNKNOWN_{}>>").format(idx)

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

            logger.debug("マスキング適用", original=entity.text, mask_token=mask_token, category=category)

        return masked_text, entity_mapping, debug_info
