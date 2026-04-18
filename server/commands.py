"""Stage 1 of the V2 pipeline: spoken punctuation command detection.

Runs on raw ASR output before the punct+case model. Two responsibilities:

1. Strip any ASR-supplied punctuation. The punct_case model
   (`oliverguhr/fullstop-punctuation-multilingual-base`, XLM-R base) was
   trained on lowercased, unpunctuated text — giving it pre-punctuated
   input puts out-of-distribution tokens in its context window. Moonshine
   rarely emits punctuation today, but we strip defensively.

2. Detect + substitute spoken punctuation/layout commands. Users dictate
   "hello comma how are you" and expect "hello, how are you?". Detection
   is rule-based (positional + lexical); the hard problem is deciding
   whether a word is the *command* or the *literal word* ("use a comma
   here"). The default leans toward command — the plan accepts occasional
   false positives since users can correct them.

See `punctuation-service-v2-plan.md` § Stage 1.
"""

from __future__ import annotations

import re


PUNCT_COMMANDS: dict[str, str] = {
    "comma": ",",
    "period": ".",
    "full stop": ".",
    "question mark": "?",
    "exclamation point": "!",
    "exclamation mark": "!",
    "colon": ":",
    "semicolon": ";",
    "dash": "—",
    "hyphen": "-",
    "open quote": "\"",
    "close quote": "\"",
    "open parenthesis": "(",
    "close parenthesis": ")",
    "ellipsis": "…",
}

# Layout commands (`new line`, `new paragraph`) are defined here for
# reference but intentionally *not* in `_ALL_COMMANDS` yet — Commit 2
# of the spoken-commands workstream will wire them into segment
# splitting on the server side, emitting one stage event per resulting
# paragraph with its own deterministic `segment_id`. Emitting literal
# `\n\n` in-text is a dead end because Stage 2 (`_run_v1_pipeline`)
# tokenizes on all whitespace and would drop them silently.
LAYOUT_COMMANDS: dict[str, str] = {
    "new line": "\n",
    "new paragraph": "\n\n",
}

# Phrases that are almost never literal in dictated speech. Unlike
# "comma"/"period" which have common non-command uses, "question mark"
# or "exclamation point" in running prose is vanishingly rare.
HIGH_CONFIDENCE_COMMANDS: set[str] = {
    "exclamation point",
    "exclamation mark",
    "question mark",
}

# Multi-word phrases first so greedy matching picks "full stop" before
# considering "full" + "stop" separately.
_ALL_COMMANDS: list[tuple[str, str]] = sorted(
    PUNCT_COMMANDS.items(),
    key=lambda kv: -len(kv[0].split()),
)

_ARTICLES = {"a", "an", "the"}

# Punctuation characters we strip from raw ASR input. Kept deliberately
# conservative: only sentence-boundary marks that the punct_case model
# would itself predict. Apostrophes (`don't`, `it's`) and hyphens (`co-
# op`) stay because they're word-internal and the model never sees them
# predicted as separate marks.
_STRIP_ASR_PUNCT_RE = re.compile(r"[.,?!;:\"]")


def strip_asr_punctuation(text: str) -> str:
    """Remove ASR-supplied sentence punctuation before Stage 1 / Stage 2."""
    return _STRIP_ASR_PUNCT_RE.sub("", text)


def apply_spoken_commands(text: str) -> str:
    """Full Stage 1: strip ASR punctuation, detect + substitute spoken
    punctuation/layout commands. Output may contain `,`, `.`, `?`, `\\n`,
    etc. where the user explicitly asked for them; Stage 2 fills the
    rest."""
    if not text or not text.strip():
        return text
    cleaned = strip_asr_punctuation(text)
    return _substitute_commands(cleaned)


def _substitute_commands(text: str) -> str:
    """Token-level command substitution. Assumes input is already
    stripped of ASR punctuation — attaching punctuation to tokens would
    break exact-match detection (`"comma,"` != `"comma"`)."""
    tokens = text.lower().split()
    if not tokens:
        return text
    out: list[str] = []
    i = 0
    while i < len(tokens):
        matched: tuple[str, str, int] | None = None
        for phrase, replacement in _ALL_COMMANDS:
            phrase_toks = phrase.split()
            n = len(phrase_toks)
            if tokens[i : i + n] != phrase_toks:
                continue
            if not _is_command_context(tokens, i, n, phrase):
                continue
            matched = (phrase, replacement, n)
            break
        if matched is None:
            out.append(tokens[i])
            i += 1
            continue
        phrase, replacement, n = matched
        if out:
            # Glue punctuation to the preceding word: "hello ," → "hello,".
            out[-1] = out[-1] + replacement
        else:
            # Utterance starts with a punctuation command. Rare but
            # don't crash — just prepend standalone.
            out.append(replacement)
        i += n
    return " ".join(out)


def _is_command_context(tokens: list[str], i: int, n: int, phrase: str) -> bool:
    """Decide whether the phrase at `tokens[i:i+n]` is a command or the
    literal word. Errs toward command per the plan; false positives are
    visible and user-correctable, while false negatives (missed commands)
    are silently frustrating. Tuning knob — revisit once we have an eval
    set with labeled command vs. literal uses."""
    if phrase in HIGH_CONFIDENCE_COMMANDS:
        return True
    # Standalone-ish short utterance: "period", "comma now", "just comma".
    if len(tokens) <= 2:
        return True
    # Preceded by an article → almost certainly literal ("use a comma",
    # "the period of history").
    if i > 0 and tokens[i - 1] in _ARTICLES:
        return False
    return True
