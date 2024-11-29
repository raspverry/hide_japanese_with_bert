import statistics
import time
from enum import Enum
from pathlib import Path

import pandas as pd
import requests
import typer


class Environment(str, Enum):
	DOCKER = "docker"
	LOCAL = "local"


class TextSizePreset(str, Enum):
	SMALL = "small"  # [1, 2, 5]
	MEDIUM = "medium"  # [1, 2, 5, 10, 20]
	LARGE = "large"  # [1, 2, 5, 10, 20, 50]
	CUSTOM = "custom"  # ユーザー指定


# Define module-level default values
DEFAULT_CATEGORIES = ["ORG", "PERSON", "LOCATION"]
DEFAULT_OUTPUT_DIR = Path("benchmark_results")
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_ITERATIONS = 5
DEFAULT_SIZE_PRESET = TextSizePreset.MEDIUM
DEFAULT_ENV_OPTION = typer.Option(..., help="実行環境 (docker/local)")


app = typer.Typer(help="GPU性能ベンチマークツール")


def generate_test_data(size_multiplier: int) -> str:
	"""テスト用テキストデータの生成"""
	base_text = (
		"最先端アルゴリズムの社会実装に取り組むAIスタートアップ、"
		"株式会社Lightblue(代表取締役:園田亜斗夢、本社:東京都千代田区、"
		"いか「Lightblue」)は、生成AIの導入効果を最大化するための"
		"診断サービス「RAG Ready診断」をリリースいたしました。"
	)
	return base_text * size_multiplier


def get_size_list(preset: TextSizePreset, custom_sizes: str | None = None) -> list[int]:
	"""テストサイズリストの取得"""
	preset_sizes = {
		TextSizePreset.SMALL: [1, 2, 5],
		TextSizePreset.MEDIUM: [1, 2, 5, 10, 20],
		TextSizePreset.LARGE: [1, 2, 5, 10, 20, 50],
	}

	if preset == TextSizePreset.CUSTOM:
		if not custom_sizes:
			raise ValueError("カスタムサイズが指定されていません")
		try:
			return [int(x.strip()) for x in custom_sizes.split(",")]
		except ValueError as e:
			raise ValueError("無効なカスタムサイズフォーマット。例: '1,2,5,10'") from e

	return preset_sizes[preset]


def run_benchmark(
	api_url: str, sizes: list[int], iterations: int, categories: list[str]
) -> dict:
	"""ベンチマークの実行"""
	results = {"text_sizes": [], "avg_times": [], "std_devs": [], "total_times": []}

	for size in sizes:
		test_text = generate_test_data(size)
		times = []

		typer.echo(f"\nテストサイズ: {size}x 基本テキスト")

		with typer.progressbar(range(iterations), label="テスト実行中") as progress:
			for i in progress:
				start_time = time.time()

				response = requests.post(
					f"{api_url}/mask_text",
					json={"text": test_text, "categories_to_mask": categories},
				)

				if response.status_code != 200:
					typer.secho(f"エラー: {response.status_code}", fg=typer.colors.RED)
					continue

				end_time = time.time()
				processing_time = end_time - start_time
				times.append(processing_time)

				typer.echo(f"反復 {i+1}: {processing_time:.3f}秒")

		avg_time = statistics.mean(times)
		std_dev = statistics.stdev(times) if len(times) > 1 else 0
		total_time = sum(times)

		results["text_sizes"].append(size)
		results["avg_times"].append(avg_time)
		results["std_devs"].append(std_dev)
		results["total_times"].append(total_time)

		typer.echo(f"\nサイズ {size}x の結果:")
		typer.echo(f"平均処理時間: {avg_time:.3f}秒")
		typer.echo(f"標準偏差: {std_dev:.3f}秒")
		typer.echo(f"合計処理時間: {total_time:.3f}秒")

	return results


def save_results(results: dict, environment: str, output_dir: Path) -> Path:
	"""結果をCSVとして保存"""
	df = pd.DataFrame(
		{
			"Environment": [environment] * len(results["text_sizes"]),
			"Text_Size_Multiplier": results["text_sizes"],
			"Average_Time": results["avg_times"],
			"Standard_Deviation": results["std_devs"],
			"Total_Time": results["total_times"],
		}
	)

	output_dir.mkdir(parents=True, exist_ok=True)
	timestamp = time.strftime("%Y%m%d_%H%M%S")
	filename = output_dir / f"benchmark_results_{environment.lower()}_{timestamp}.csv"
	df.to_csv(filename, index=False)
	return filename


@app.command()
def run(
	environment: Environment = DEFAULT_ENV_OPTION,
	size_preset: TextSizePreset = TextSizePreset.MEDIUM,
	custom_sizes: str | None = None,
	iterations: int = DEFAULT_ITERATIONS,
	api_url: str = DEFAULT_API_URL,
	output_dir: Path = DEFAULT_OUTPUT_DIR,
	categories: list[str] = DEFAULT_CATEGORIES,
):
	"""GPUベンチマークを実行し、結果をCSVファイルとして保存します"""

	typer.echo(f"\n{environment.value.upper()} 環境でベンチマークを開始...")

	try:
		sizes = get_size_list(size_preset, custom_sizes)
		results = run_benchmark(api_url, sizes, iterations, categories)
		results_file = save_results(results, environment.value, output_dir)
		typer.secho(
			f"\nベンチマークが完了しました。結果は {results_file} で確認できます。",
			fg=typer.colors.GREEN,
		)
	except Exception as e:
		typer.secho(f"エラーが発生しました: {str(e)}", fg=typer.colors.RED)
		raise typer.Exit(code=1) from e


if __name__ == "__main__":
	app()
