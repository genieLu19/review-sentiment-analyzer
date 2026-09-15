from sentiment_analyzer.preprocess import clean_text


def test_clean_text_lowercases_and_removes_punctuation() -> None:
    assert clean_text("Очень ХОРОШАЯ книга!!!") == "очень хорошая книга"


def test_clean_text_replaces_yo_letter() -> None:
    assert clean_text("Тяжёлый день") == "тяжелый день"


def test_clean_text_handles_none() -> None:
    assert clean_text(None) == ""
