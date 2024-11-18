# app/logger_config.py

import json
import logging
import sys
from logging.handlers import RotatingFileHandler

import structlog


def configure_logging():
	"""
	ロギング設定を構成します。
	コンソールには読みやすい形式で、ファイルにはJSON形式でログを出力します。
	"""
	# ログレベルの設定（デフォルトはINFO）
	LOG_LEVEL = logging.INFO

	# ログファイルの設定
	LOG_FILE = "log/app.log"
	LOG_ROTATION = 5  # ログファイルのバックアップ数

	# 既存のハンドラーをクリア
	root_logger = logging.getLogger()
	for handler in root_logger.handlers[:]:
		root_logger.removeHandler(handler)
	root_logger.setLevel(logging.NOTSET)

	# structlogの設定
	structlog.configure(
		processors=[
			structlog.stdlib.filter_by_level,  # ログレベルでフィルタリング
			structlog.stdlib.add_logger_name,  # ロガー名を追加
			structlog.stdlib.add_log_level,  # ログレベルを追加
			structlog.processors.TimeStamper(fmt="iso"),  # タイムスタンプをISO形式で追加
			# ProcessorFormatter用のラッパー
			structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
		],
		context_class=dict,
		logger_factory=structlog.stdlib.LoggerFactory(),
		wrapper_class=structlog.stdlib.BoundLogger,
		cache_logger_on_first_use=True,
		# disable_existing_loggers=True,  # この行を削除
	)

	# コンソールハンドラーの設定（読みやすい形式）
	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setLevel(LOG_LEVEL)
	formatter_console = structlog.stdlib.ProcessorFormatter(
		processor=structlog.dev.ConsoleRenderer(),
		foreign_pre_chain=[
			structlog.processors.TimeStamper(fmt="iso"),
		],
	)
	console_handler.setFormatter(formatter_console)

	# ファイルハンドラーの設定（JSON形式、ensure_ascii=False）
	formatter_file = structlog.stdlib.ProcessorFormatter(
		processor=lambda logger, name, event_dict: json.dumps(event_dict, ensure_ascii=False),
		foreign_pre_chain=[
			structlog.processors.TimeStamper(fmt="iso"),
		],
	)

	file_handler = RotatingFileHandler(
		LOG_FILE, maxBytes=10**6, backupCount=LOG_ROTATION, encoding="utf-8"
	)
	file_handler.setLevel(LOG_LEVEL)
	file_handler.setFormatter(formatter_file)

	# ルートロガーにハンドラーを追加
	root_logger.addHandler(console_handler)
	root_logger.addHandler(file_handler)


# ロギング設定を実行
configure_logging()
