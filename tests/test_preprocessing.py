from src.preprocessing import tokenize_for_bm25


def test_tokenize_lowercases_and_splits():
    out = tokenize_for_bm25("CO2 emissions Increase!")
    assert out == ["co2", "emissions", "increase"]


def test_tokenize_drops_pure_punct_and_short_noise():
    out = tokenize_for_bm25("--- a, the of climate.")
    assert "climate" in out
    assert "a" not in out and "the" not in out and "of" not in out


def test_tokenize_handles_empty():
    assert tokenize_for_bm25("") == []
