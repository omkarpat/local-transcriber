"""
Minimal FastAPI stub for local iOS development of the punctuation client.
Matches the API contract in punctuation-service-plan.md; returns a
visually-distinct "punctuated" version of the input (first letter
uppercased + trailing period) so the round-trip is obvious in the UI.

Not the real model — just enough surface area to verify the iOS-side
plumbing (timeouts, fallback, upsert-in-place) before the real server
is deployed.

Run:
    pip install fastapi uvicorn
    python tools/punctuation_stub.py
    # listens on http://localhost:8000/punctuate

iOS side (default config) points at localhost:8000 and sends an
X-API-Key header; we intentionally don't check it here so you can also
exercise the "missing key" path if needed by flipping the
`REQUIRE_API_KEY` flag below.

Flip behaviors for testing the iOS fallback paths:
    DELAY_SECONDS  -> sleep before responding (> 2.0 to force client timeout)
    FAIL_STATUS    -> return this HTTP status instead of 200
    MALFORMED      -> return a response missing the `text` key
"""

import time
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn

DELAY_SECONDS = 0.0
FAIL_STATUS: int | None = None
MALFORMED = False
REQUIRE_API_KEY = False
EXPECTED_API_KEY = "dev-key-change-me"

app = FastAPI()


class PunctuateRequest(BaseModel):
    text: str


@app.post("/punctuate")
async def punctuate(body: PunctuateRequest, request: Request):
    if REQUIRE_API_KEY:
        got = request.headers.get("X-API-Key", "")
        if got != EXPECTED_API_KEY:
            raise HTTPException(status_code=401, detail="missing or invalid API key")

    if DELAY_SECONDS > 0:
        time.sleep(DELAY_SECONDS)

    if FAIL_STATUS is not None:
        raise HTTPException(status_code=FAIL_STATUS, detail="forced failure for testing")

    text = body.text.strip()
    if not text:
        return {"text": ""}

    punctuated = text[0].upper() + text[1:]
    if not punctuated.endswith((".", "?", "!")):
        punctuated += "."

    if MALFORMED:
        return {"wrong_key": punctuated}

    return {"text": punctuated}


@app.get("/healthz")
def healthz():
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
