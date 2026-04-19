"""Unit tests for `commands.py` (Stage 1 of the V2 pipeline).

Pure functions — no model load required. Run these on every change to
the command detector as a cheap regression gate."""

from __future__ import annotations

import pytest

from commands import (
    apply_spoken_commands,
    strip_asr_punctuation,
)


class TestStripAsrPunctuation:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("hello, world", "hello world"),
            ("hello. world", "hello world"),
            ("what? now", "what now"),
            ("no! way", "no way"),
            ("wait: then", "wait then"),
            ("wait; then", "wait then"),
            ('she said "hi"', "she said hi"),
            # Word-internal marks preserved.
            ("don't strip", "don't strip"),
            ("co-op", "co-op"),
            # No-ops.
            ("clean text", "clean text"),
            ("", ""),
        ],
    )
    def test_strip(self, raw: str, expected: str) -> None:
        assert strip_asr_punctuation(raw) == expected


class TestBasicSubstitution:
    """Happy-path: each command appears in an unambiguous context and fires.
    `apply_spoken_commands` returns a list of paragraphs — for inputs
    without `new paragraph` the list is length 1."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("hello comma how are you", ["hello, how are you"]),
            ("send me ten dollars period", ["send me ten dollars."]),
            ("is this working question mark", ["is this working?"]),
            ("yes exclamation point", ["yes!"]),
            ("we are done full stop", ["we are done."]),
            ("this is it colon and then", ["this is it: and then"]),
            ("wait semicolon then go", ["wait; then go"]),
        ],
    )
    def test_punct_commands(self, raw: str, expected: list[str]) -> None:
        assert apply_spoken_commands(raw) == expected

    def test_new_line_still_deferred(self) -> None:
        """`new line` still has no effect in Stage 1 — Stage 2's tokenizer
        would drop a literal `\\n`, and the UI renders one segment as
        one paragraph, so there's nowhere useful to surface a soft line
        break today. The phrase survives as literal words."""
        assert apply_spoken_commands("line one new line line two") == [
            "line one new line line two"
        ]


class TestLiteralContexts:
    """Command words used literally should NOT be substituted."""

    @pytest.mark.parametrize(
        "raw",
        [
            "use a comma here to make it clearer",
            "the period of history was interesting",
            "place a dash between those two dates",
        ],
    )
    def test_preceded_by_article(self, raw: str) -> None:
        result = apply_spoken_commands(raw)
        # Single-paragraph result — no literal contexts trigger layout splits.
        assert len(result) == 1
        text = result[0]
        # None of the punctuation marks should have been inserted.
        for mark in (",", ".", ":", ";", "—", "?", "!"):
            assert mark not in text, f"{mark!r} leaked into {text!r}"
        # The literal word should survive.
        original_tokens = raw.split()
        for tok in ("comma", "period", "dash"):
            if tok in original_tokens:
                assert tok in text.split()


class TestHighConfidenceCommands:
    """`question mark`, `exclamation point` should fire even mid-sentence —
    they're almost never literal in dictation."""

    def test_question_mark_mid_sentence(self) -> None:
        # "Does this work question mark and if so…" — question mark fires.
        result = apply_spoken_commands("does this work question mark and if so do that")
        assert len(result) == 1
        assert "?" in result[0]

    def test_exclamation_point_mid_sentence(self) -> None:
        result = apply_spoken_commands("amazing exclamation point look at this")
        assert len(result) == 1
        assert "!" in result[0]


class TestShortUtterances:
    """≤2 tokens: whatever matches fires, regardless of article rule."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("period", ["."]),
            ("comma", [","]),
            ("full stop", ["."]),
            ("question mark", ["?"]),
        ],
    )
    def test_standalone(self, raw: str, expected: list[str]) -> None:
        assert apply_spoken_commands(raw) == expected


class TestAsrPunctuationOverlap:
    """Scenarios where the ASR supplies punctuation on top of speech."""

    def test_asr_comma_stripped(self) -> None:
        # Strip happens before detection; no command word present →
        # output is clean, model fills in later.
        assert apply_spoken_commands("hello, how are you") == ["hello how are you"]

    def test_asr_punct_plus_redundant_command(self) -> None:
        # ASR inserted a comma AND user said "comma". Strip removes the
        # ASR comma, command inserts one back. Net: single comma, no dupe.
        assert apply_spoken_commands("hello, comma how are you") == ["hello, how are you"]

    def test_asr_period_end_stripped(self) -> None:
        # ASR-style trailing period is stripped; model backfills in stage 2.
        assert apply_spoken_commands("hello world.") == ["hello world"]


class TestEdgeCases:
    def test_empty_string(self) -> None:
        # Empty input returns a single-element list so callers can rely on
        # `len(result) >= 1` (they iterate stage events per segment).
        assert apply_spoken_commands("") == [""]

    def test_whitespace_only(self) -> None:
        # Whitespace-only passes through untouched in its sole element.
        assert apply_spoken_commands("   ") == ["   "]

    def test_single_word_not_command(self) -> None:
        assert apply_spoken_commands("hello") == ["hello"]

    def test_multi_word_command_greedy(self) -> None:
        # "full stop" should match as a 2-token command, not "full" + "stop".
        assert apply_spoken_commands("done full stop") == ["done."]

    def test_command_at_utterance_start(self) -> None:
        # Leading command gets prepended standalone (rare edge case).
        assert apply_spoken_commands("comma then continue") == [", then continue"]


class TestNewParagraphSplitting:
    """`new paragraph` closes the current paragraph and starts a new one.
    It's HIGH_CONFIDENCE so it fires even mid-utterance and after
    articles ("a new paragraph here" → splits; false positive is
    accepted per plan)."""

    def test_splits_into_two(self) -> None:
        assert apply_spoken_commands("hello new paragraph world") == [
            "hello", "world"
        ]

    def test_punct_before_split_glues(self) -> None:
        # "comma" glues to preceding word of paragraph 1; split follows.
        assert apply_spoken_commands("hello comma new paragraph world") == [
            "hello,", "world"
        ]

    def test_three_paragraphs(self) -> None:
        assert apply_spoken_commands(
            "hello new paragraph world new paragraph again"
        ) == ["hello", "world", "again"]

    def test_leading_split_drops_empty_first(self) -> None:
        # Split before any words → empty first paragraph dropped; result
        # is a single paragraph with the trailing text.
        assert apply_spoken_commands("new paragraph hello world") == ["hello world"]

    def test_trailing_split_drops_empty_last(self) -> None:
        assert apply_spoken_commands("hello world new paragraph") == ["hello world"]

    def test_only_new_paragraph(self) -> None:
        # Degenerate input — all paragraphs empty. Preserve the
        # "always return >= 1 element" invariant with a single empty
        # string so the server still emits one final event.
        assert apply_spoken_commands("new paragraph") == [""]

    def test_after_article_still_splits(self) -> None:
        # HIGH_CONFIDENCE bypasses the article rule — accepted false
        # positive per plan (users can correct, and literal uses of
        # "new paragraph" in prose are vanishingly rare).
        assert apply_spoken_commands("here is a new paragraph section") == [
            "here is a", "section"
        ]
