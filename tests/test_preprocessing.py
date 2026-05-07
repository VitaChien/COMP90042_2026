from src.preprocessing import SYNONYM_MAP, tokenize_for_bm25


def test_tokenize_lowercases_and_stems():
    out = tokenize_for_bm25("Global Warming Increases!")
    # 'global' -> 'global', 'warming' -> 'warm', 'increases' -> 'increas'
    assert out == ["global", "warm", "increas"]


def test_tokenize_drops_stopwords_and_short_noise():
    out = tokenize_for_bm25("--- a, the of climate.")
    # 'climate' stems to 'climat'; stopwords/single-char tokens dropped
    assert out == ["climat"]


def test_tokenize_handles_empty():
    assert tokenize_for_bm25("") == []


def test_tokenize_preserves_year_tokens():
    out = tokenize_for_bm25("In 2021 emissions rose.")
    # year stays as-is (digits bypass stopword/length filter and stemmer)
    assert "2021" in out


def test_synonym_expansion_co2():
    out = tokenize_for_bm25("CO2 levels are rising")
    # 'co2' expands to itself + 'carbon dioxide', then everything is stemmed
    assert "co2" in out
    assert "carbon" in out
    assert "dioxid" in out  # 'dioxide' stems to 'dioxid'


def test_synonym_expansion_ch4():
    out = tokenize_for_bm25("CH4 emissions")
    assert "ch4" in out
    assert "methan" in out  # 'methane' stems to 'methan'


def test_synonym_expansion_n2o():
    out = tokenize_for_bm25("N2O concentrations")
    assert "n2o" in out
    assert "nitrous" in out
    assert "oxid" in out  # 'oxide' stems to 'oxid'


def test_synonym_expansion_h2o():
    out = tokenize_for_bm25("H2O cycles")
    assert "h2o" in out
    assert "water" in out


def test_synonym_expansion_ghg():
    out = tokenize_for_bm25("GHG targets")
    assert "ghg" in out
    assert "greenhous" in out  # 'greenhouse' stems to 'greenhous'
    assert "gas" in out


def test_synonym_map_covers_target_abbreviations():
    # Sanity: regression check that the map still includes the high-value
    # climate abbreviations the data analysis flagged.
    for key in ("co2", "ch4", "n2o", "h2o", "ghg"):
        assert key in SYNONYM_MAP


def test_stem_collapses_warming_variants():
    # All four surface forms should stem to a single token, enabling BM25
    # to match across morphological variants.
    stems = {tokenize_for_bm25(w)[0] for w in ("warming", "warmed", "warms", "warm")}
    assert stems == {"warm"}
