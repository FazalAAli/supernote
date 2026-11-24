"""Unit tests for `supernote.server.components.search.lexical_scoring`."""

from __future__ import annotations

import pytest

from supernote.server.components.search.lexical_scoring import (
    build_lexical_preview,
    extract_centered_snippet,
    parse_lexical_query,
    score_page_lexical,
    sql_prefilter_terms,
)


@pytest.mark.parametrize("raw_query", ["", "   ", "###", "---", "の"])
def test_parse_lexical_query_rejects_empty_and_particle_only_inputs(
    raw_query: str,
) -> None:
    assert parse_lexical_query(raw_query) is None


@pytest.mark.parametrize(
    ("raw_query", "keywords", "expected_score", "expected_preview"),
    [
        (
            "#KS-Tilt",
            ("###", "unrelated", "ks-tilt", "KS-TILT"),
            1.0,
            "Keywords: #ks-tilt, #unrelated",
        ),
        ('"#ks-tilt"', ("ks-tilt",), 1.0, "Keywords: #ks-tilt"),
        ("r", ("r",), 1.0, "Keywords: #r"),
        ("now", ("now",), 1.0, "Keywords: #now"),
        (
            "#roadmap #Q4",
            ("backlog", "roadmap", "Q4"),
            0.95,
            "Keywords: #roadmap, #Q4, #backlog",
        ),
        ("tilt", ("ks-tilt",), 0.92, "Keywords: #ks-tilt"),
        ("C++", ("C++17",), 0.92, "Keywords: #C++17"),
        ("C#", ("C#12",), 0.92, "Keywords: #C#12"),
        ("$100", ("$100",), 1.0, "Keywords: #$100"),
        ("how to fix ks-tilt drift", ("ks-tilt",), 0.80, "Keywords: #ks-tilt"),
        ("trap", ("ks-trapezoid",), 0.85, "Keywords: #ks-trapezoid"),
        ("日本語の勉強", ("日本語の勉強メモ",), 0.92, "Keywords: #日本語の勉強メモ"),
        ("Python入門", ("Python入門",), 1.0, "Keywords: #Python入門"),
        ("人々", ("人々の暮らし",), 0.92, "Keywords: #人々の暮らし"),
    ],
)
def test_score_page_lexical_native_keywords_and_preview(
    raw_query: str,
    keywords: tuple[str, ...],
    expected_score: float,
    expected_preview: str,
) -> None:
    parsed = parse_lexical_query(raw_query)
    assert parsed is not None
    assert score_page_lexical(parsed, keywords, None) == pytest.approx(expected_score)
    assert build_lexical_preview(keywords, None, parsed) == expected_preview


@pytest.mark.parametrize(
    ("raw_query", "keywords", "text_content"),
    [
        ("a", ("ks-tilt",), "This is a note about everything."),
        ("the", ("ks-tilt",), "The quick brown fox."),
        ("1", ("ks-tilt",), "Page 1 item 1."),
        ("$a", ("ks-tilt",), "Variable a equals 1."),
        ("r", ("roadmap",), "R statistical computing notes."),
        ("what is on", ("on",), "What is going on in this meeting?"),
        ("to be or not to be", ("not",), "This page might not be ready yet."),
        ("x-ray", ("ray-tracing",), "A bright ray of sunshine."),
        ("p-value", ("business-value",), "Great resale value."),
        ("b-tree", ("family-tree",), "Planted an oak tree."),
        ("vitamin d", ("vitamin-c",), "Take vitamin B12 daily."),
        ("c programming", ("python-programming",), "Python programming guide."),
        ("Python入門", (), "料理入門のレシピ。"),
        ("3月15日", (), "12月31日の予定。"),
        ("100円", (), "合計50000円です。"),
        ("人々", (), "日本人のパスポート。"),
        ("#ks-tilt", (), "ks is completely unrelated to tilt here."),
        ("ks-tilt", ("ks",), None),
        ("machine learning", ("sewing-machine", "spanish-learning"), None),
        ("cat", ("vacation",), None),
        ("cat 日本語", ("vacation 日本語",), "日本語 on my vacation"),
        ("日本語の物理", (), "毎日日本語を勉強する。"),
        ("C#", ("C++",), "Modern C++ reference."),
        ("C++", ("Vitamin C",), "Daily vitamin C log."),
        ("$100", (), "Counted 100 resistors in bin."),
        ("$10", ("$100",), "Budget is $100 total."),
    ],
)
def test_score_page_lexical_rejects_false_positives(
    raw_query: str,
    keywords: tuple[str, ...],
    text_content: str | None,
) -> None:
    parsed = parse_lexical_query(raw_query)
    assert parsed is not None
    assert score_page_lexical(parsed, keywords, text_content) == 0.0


@pytest.mark.parametrize(
    ("raw_query", "keywords", "text_content", "expected_score", "expected_preview"),
    [
        (
            "telescope alignment?",
            (),
            "Checked telescope alignment last night.",
            0.85,
            "Checked telescope alignment last night.",
        ),
        (
            "cost",
            (),
            "Calculated cost+tax=100 total.",
            0.85,
            "Calculated cost+tax=100 total.",
        ),
        ("issue", (), "Fixed issue#42 yesterday.", 0.85, "Fixed issue#42 yesterday."),
        (
            "C++",
            (),
            "Modern C++17 templates and concepts.",
            0.85,
            "Modern C++17 templates and concepts.",
        ),
        ("C#", (), "Upgraded C#12 compiler.", 0.85, "Upgraded C#12 compiler."),
        ("Python", (), "Python入門ノート。", 0.85, "Python入門ノート。"),
        ("3月15日", (), "次回は3月15日に開催。", 0.85, "次回は3月15日に開催。"),
        (
            "日本語を勉強する",
            (),
            "今日は日本語の勉強をした。",
            0.75,
            "今日は日本語の勉強をした。",
        ),
        ("机器学习", (), "深度学习与机器学习笔记。", 0.85, "深度学习与机器学习笔记。"),
        ("straße", (), "Berliner Straße 10.", 0.85, "Berliner Straße 10."),
        (
            "alignment mirror",
            (),
            "Primary mirror needs careful laser alignment.",
            0.75,
            "Primary mirror needs careful laser alignment.",
        ),
        (
            "optics",
            ("optics-lab",),
            "Optics bench collimation.",
            0.92 + 0.05 * 0.85,
            "[#optics-lab] Optics bench collimation.",
        ),
    ],
)
def test_score_page_lexical_transcript_and_combined_boost(
    raw_query: str,
    keywords: tuple[str, ...],
    text_content: str,
    expected_score: float,
    expected_preview: str,
) -> None:
    parsed = parse_lexical_query(raw_query)
    assert parsed is not None
    assert score_page_lexical(parsed, keywords, text_content) == pytest.approx(
        expected_score
    )
    assert build_lexical_preview(keywords, text_content, parsed) == expected_preview


def test_sql_prefilter_terms_caps_depth_separates_cjk_and_skips_casefold_expansions() -> (
    None
):
    many_words = " ".join(f"term{i:04d}" for i in range(1200))
    dup_query = f"term0001 term0001 {many_words} 日本語 $"
    parsed_many = parse_lexical_query(dup_query)
    assert parsed_many is not None
    ascii_terms, cjk_terms = sql_prefilter_terms(parsed_many, dup_query)
    assert len(ascii_terms) + len(cjk_terms) == 8

    parsed_cjk = parse_lexical_query("Python 日本語")
    assert parsed_cjk is not None
    assert sql_prefilter_terms(parsed_cjk, "Python 日本語") == (
        ("python",),
        ("日本語",),
    )

    parsed_eszett = parse_lexical_query("Straße")
    assert parsed_eszett is not None
    assert sql_prefilter_terms(parsed_eszett, "Straße") == ((), ())


def test_extract_centered_snippet_centers_windows_and_clamps_casefold_expansions() -> (
    None
):
    head_long = "Optics alignment header. " + ("Routine text. " * 20)
    assert extract_centered_snippet(head_long, ["", "optics"]).endswith("…")

    tail_long = ("Routine standup notes. " * 12) + "Parabolic mirror focal test."
    tail_snip = extract_centered_snippet(tail_long, ["parabolic"])
    assert tail_snip.startswith("…")
    assert tail_snip.endswith("Parabolic mirror focal test.")

    expanded_prefix = ("ß" * 200) + " parabolic"
    clamped_direct = extract_centered_snippet(expanded_prefix, ["parabolic"])
    assert clamped_direct.startswith("…")
    assert clamped_direct.endswith(" parabolic")
    assert len(clamped_direct) <= 200

    expanded_folded = ("ß" * 200) + " Straße"
    clamped_folded = extract_centered_snippet(expanded_folded, ["strasse"])
    assert clamped_folded.startswith("…")
    assert clamped_folded.endswith(" Straße")
    assert len(clamped_folded) <= 200


def test_build_lexical_preview_returns_empty_when_page_has_no_text_or_tags() -> None:
    parsed = parse_lexical_query("optics")
    assert parsed is not None
    assert build_lexical_preview((), None, parsed) == ""
