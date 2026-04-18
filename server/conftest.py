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
    helper that runs the full V2 pipeline on raw input. The per-stage
    dict lets individual tests assert on whichever stage is relevant
    without re-running the expensive stages."""
    import app
    from commands import apply_spoken_commands

    app._load_model()
    app._load_itn()

    def run(text: str) -> dict[str, str]:
        stage1 = apply_spoken_commands(text)
        stage2 = app._run_v1_pipeline(stage1)
        stage3 = app._stage_itn(stage2)
        return {"raw": text, "stage1": stage1, "stage2": stage2, "stage3": stage3}

    return run
