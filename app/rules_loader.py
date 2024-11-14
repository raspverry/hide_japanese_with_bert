# app/rules_loader.py

import os
import json
import re
from typing import Dict, List, Pattern, Set, Tuple
from collections import defaultdict

import structlog

from app.models import Entity

# ロガーの取得
logger = structlog.get_logger(__name__)

class RuleBasedMasker:
    """ルールベースのマスキング処理を行うクラス"""

    def __init__(self, rules_file: str):
        """JSONファイルからルールを読み込む"""
        if not os.path.exists(rules_file):
            logger.error("ルールファイルが見つかりません", rules_file=rules_file)
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

    def _compile_patterns(self) -> Dict[str, List[Pattern]]:
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
        # コンパイルされたパターンをログに記録
        # str(p)ではなくp.patternを使用してパターン文字列のみを取得
        compiled_patterns_str = [
            p.pattern for p_list in compiled.values()
            for p in (p_list if isinstance(p_list, list) else [item for sublist in p_list.values() for item in sublist])
        ]
        logger.debug("コンパイルされたパターン", compiled_patterns=compiled_patterns_str)
        return compiled

    def _is_excluded(self, text: str) -> bool:
        """除外パターンに該当するかチェック"""
        exclusions = self.rules['exclusions']

        # 共通単語チェック
        if text in exclusions['common_words']:
            logger.debug("除外対象の共通単語", text=text)
            return True

        # 安全なパターンチェック
        for pattern in exclusions['safe_patterns']:
            if re.search(pattern, text, re.UNICODE | re.IGNORECASE):
                logger.debug("除外対象の安全パターンに一致", text=text, pattern=pattern)
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
                                logger.debug("マッチ検出", text=match.group(), category=category, start=start, end=end)
            elif isinstance(patterns, dict):
                for sub_category, sub_patterns in patterns.items():
                    for pattern in sub_patterns:
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
                                    logger.debug("マッチ検出", text=match.group(), category=category, start=start, end=end)

        logger.debug("マッチ検出結果", matches=[e.__dict__ for e in matches])
        return sorted(matches, key=lambda x: x.start)
