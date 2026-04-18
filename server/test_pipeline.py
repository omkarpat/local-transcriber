"""Integration tests for the full V2 streaming pipeline (commands →
punct_case → itn). Loads the ONNX model + NeMo ITN once per session
via the `pipeline` fixture in `conftest.py`.

Assertions are property-based rather than exact-match where the
model's output could drift (quantization, chunking, future model
swaps). For Stage 1's output we *can* assert exact strings — that's
a pure function of `commands.py` and `test_commands.py` already
covers it — so here we focus on Stage 2 and 3 behavior.

Run this file specifically: `pytest test_pipeline.py`
Skip it:                    `pytest -m "not model"`
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.model


class TestCommandSubstitutionSurvives:
    """User-inserted marks from Stage 1 should pass through Stage 2 and 3."""

    def test_spoken_comma_reaches_final(self, pipeline) -> None:
        out = pipeline("hello comma how are you")
        assert "," in out["stage3"]
        # Command word itself must not leak into the rendered transcript.
        assert "comma" not in out["stage3"].lower()

    def test_spoken_period_reaches_final(self, pipeline) -> None:
        out = pipeline("we are done period")
        assert "." in out["stage3"]
        assert "period" not in out["stage3"].lower()

    def test_question_mark_reaches_final(self, pipeline) -> None:
        out = pipeline("is this working question mark")
        assert "?" in out["stage3"]
        assert "question mark" not in out["stage3"].lower()

    def test_full_stop_reaches_final(self, pipeline) -> None:
        out = pipeline("we are done full stop")
        assert "." in out["stage3"]
        assert "full stop" not in out["stage3"].lower()


class TestAsrSuppliedPunctuation:
    """Real scenario: Moonshine sometimes emits punctuation. Stage 1
    strips it so the model sees its native training format."""

    def test_asr_comma_does_not_duplicate(self, pipeline) -> None:
        # Stage 1 strips the `,`; model may add one back. Either way,
        # no double-comma artifact should reach the final stage.
        out = pipeline("hello, how are you")
        assert ",," not in out["stage3"]

    def test_asr_period_does_not_duplicate(self, pipeline) -> None:
        out = pipeline("we are done.")
        assert ".." not in out["stage3"]

    def test_asr_punct_plus_redundant_command_single_mark(self, pipeline) -> None:
        # ASR added `,` AND user said "comma" — net single comma.
        out = pipeline("hello, comma how are you")
        assert ",," not in out["stage3"]
        assert "," in out["stage3"]


class TestLiteralWordPassthrough:
    """Command words used as literals (preceded by article) must survive
    as plain words through all three stages."""

    def test_literal_comma_survives(self, pipeline) -> None:
        out = pipeline("use a comma here to make it clearer")
        assert "comma" in out["stage3"].lower()

    def test_literal_period_survives(self, pipeline) -> None:
        out = pipeline("the period of history was interesting")
        assert "period" in out["stage3"].lower()


class TestItnRunsOnStage1Output:
    """Numbers should still get normalized even when Stage 1 inserted
    punctuation earlier in the sentence."""

    def test_currency_after_command(self, pipeline) -> None:
        out = pipeline("i owe you twenty dollars comma please pay me")
        assert "$20" in out["stage3"]
        assert "," in out["stage3"]

    def test_year_at_sentence_end_with_period(self, pipeline) -> None:
        out = pipeline("let's plan for twenty twenty seven period")
        assert "2027" in out["stage3"]
        assert "." in out["stage3"]


class TestCasing:
    """Stage 2's casing pass should capitalize sentence starts."""

    def test_first_word_capitalized(self, pipeline) -> None:
        out = pipeline("hello how are you")
        assert out["stage3"][0].isupper()


class TestNoLeaks:
    """Broad safety net: none of the command phrases should survive into
    the final rendered transcript for cases where we expect them to fire."""

    @pytest.mark.parametrize(
        "raw, forbidden_fragment",
        [
            ("hello comma world",           "comma"),
            ("hello period",                "period"),
            ("what question mark",          "question mark"),
            ("no exclamation point",        "exclamation point"),
            ("done full stop",              "full stop"),
        ],
    )
    def test_command_word_absent(
        self, pipeline, raw: str, forbidden_fragment: str
    ) -> None:
        out = pipeline(raw)
        assert forbidden_fragment not in out["stage3"].lower()


class TestRegressionSnapshots:
    """Exact-match snapshots on representative inputs. Pin these so any
    change to the pipeline (model swap, stage reorder, ITN tweak) that
    alters user-visible output requires explicit review.

    Updating a snapshot: run the failing test, eyeball the new output,
    and update the expected string here if the new behavior is desired."""

    @pytest.mark.parametrize(
        "raw, expected_stage3",
        [
            # Note: without Stage 1, the model would output
            # "Hello, how are you?" — the user-inserted early comma
            # shifts its read of the sentence toward declarative.
            # Documented as a known quality note; not worth fighting
            # the model over.
            ("hello comma how are you",          "Hello, how are you."),
            ("we are done full stop",            "We are done."),
            ("send me ten dollars period",       "Send me $10."),
            ("let's plan for twenty twenty seven period", "Let's plan for 2027."),
        ],
    )
    def test_snapshot(self, pipeline, raw: str, expected_stage3: str) -> None:
        out = pipeline(raw)
        assert out["stage3"] == expected_stage3
