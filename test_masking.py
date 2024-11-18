import pytest

from app.masking import EnhancedTextMasker


@pytest.fixture(scope="module")
def masker():
	return EnhancedTextMasker()


def test_masking_case1(masker):
	test_text = "株式会社テクノロジーズの山田太郎部長（メール：test@example.com）"
	masked_text, mapping, debug = masker.mask_text(
		test_text,
		categories=["ORG", "PERSON", "EMAIL"],
		mask_style="descriptive",
	)
	print(masked_text)
	assert "<<組織_1>" in masked_text
	assert "<<人物_2>" in masked_text
	assert "<<メール_3>" in masked_text


def test_masking_case2(masker):
	test_text = "東京都渋谷区の本社オフィス"
	masked_text, mapping, debug = masker.mask_text(
		test_text,
		categories=["ORG", "PERSON", "LOCATION", "POSITION"],
		mask_style="descriptive",
	)
	assert "<<場所_1>" in masked_text


def test_masking_case3(masker):
	test_text = "代表取締役の田中一郎氏は、Project-Xの成功を報告しました。"
	masked_text, mapping, debug = masker.mask_text(
		test_text,
		categories=["ORG", "PERSON", "LOCATION", "POSITION"],
		mask_style="descriptive",
	)
	assert "<<役職_1>" in masked_text
	assert "<<人物_2>" in masked_text
	assert "<<組織_3>" in masked_text
