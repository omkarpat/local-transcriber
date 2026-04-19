"""Shared pytest fixtures for the server test suite.

Tests live flat next to `app.py` / `commands.py` so they can import the
private pipeline helpers without packaging ceremony.

Two test files today:
- `test_commands.py` — unit tests for `commands.py`. Fast, no model.
- `test_pipeline.py` — integration tests exercising the full streaming
  pipeline. Slow (~3-5 s cold start). Marked `@pytest.mark.model`.

Run fast tests only:     pytest -m "not model"
Run everything:          pytest
Run pipeline tests only: pytest -m model
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def pipeline():
    """Load the punct_case model + NeMo ITN once per session and return a
    helper that runs the full V2 pipeline on raw input. Stage 1 returns
    a list of paragraphs (one per `new paragraph` split); stages 2 and
    3 run per paragraph.

    Returned dict keeps the historical `stage1/2/3` keys as joined
    strings (`\\n\\n` between paragraphs) so single-paragraph tests can
    keep asserting on substrings directly. `*_segs` keys expose the
    per-paragraph lists for tests that care about segment boundaries."""
    import app
    from commands import apply_spoken_commands

    app._load_model()
    app._load_itn()

    def run(text: str) -> dict:
        stage1_segs = apply_spoken_commands(text)
        stage2_segs = [app._run_v1_pipeline(p) for p in stage1_segs]
        stage3_segs = [app._stage_itn(p) for p in stage2_segs]
        return {
            "raw": text,
            "stage1_segs": stage1_segs,
            "stage2_segs": stage2_segs,
            "stage3_segs": stage3_segs,
            "stage1": "\n\n".join(stage1_segs),
            "stage2": "\n\n".join(stage2_segs),
            "stage3": "\n\n".join(stage3_segs),
        }

    return run
