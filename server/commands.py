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

# Layout commands trigger segment breaks rather than emitting in-text
# characters — Stage 2 tokenizes on all whitespace and would drop any
# literal `\n\n` we tried to pass through. `new paragraph` splits the
# utterance into separate paragraphs (each gets its own segment id
# downstream). `new line` is deferred: Stage 2 still can't preserve a
# single `\n`, and the UI renders one segment as one paragraph anyway,
# so there's nowhere to put a soft line break today.
_LAYOUT_PHRASES: tuple[str, ...] = ("new paragraph",)

# Phrases that are almost never literal in dictated speech. Unlike
# "comma"/"period" which have common non-command uses, "question mark"
# or "new paragraph" in running prose is vanishingly rare, so they
# bypass the article rule (`a new paragraph` still splits).
HIGH_CONFIDENCE_COMMANDS: set[str] = {
    "exclamation point",
    "exclamation mark",
    "question mark",
    "new paragraph",
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


def apply_spoken_commands(text: str) -> list[str]:
    """Full Stage 1: strip ASR punctuation, detect + substitute spoken
    punctuation commands, split on layout (`new paragraph`) commands.
    Returns one string per resulting paragraph — callers emit each as
    its own downstream segment. Always returns at least one element
    (possibly empty) so the caller can rely on a non-empty result."""
    if not text or not text.strip():
        return [text]
    cleaned = strip_asr_punctuation(text)
    return _substitute_commands(cleaned)


def _substitute_commands(text: str) -> list[str]:
    """Token-level command substitution with paragraph splitting.
    Assumes input is already stripped of ASR punctuation — attaching
    punctuation to tokens would break exact-match detection (`"comma,"`
    != `"comma"`). Layout commands close the current paragraph and open
    a new one; punctuation commands glue to the previous word in the
    current paragraph."""
    tokens = text.lower().split()
    if not tokens:
        return [text]
    paragraphs: list[list[str]] = [[]]
    i = 0
    while i < len(tokens):
        # Layout commands first — they don't emit a character, they break
        # the paragraph. If the phrase matches but context rules reject
        # it, fall through to word-level handling (HIGH_CONFIDENCE
        # bypasses the article rule for "new paragraph").
        layout_n = _match_layout(tokens, i)
        if layout_n is not None:
            paragraphs.append([])
            i += layout_n
            continue

        punct_match = _match_punct(tokens, i)
        if punct_match is not None:
            replacement, n = punct_match
            if paragraphs[-1]:
                # Glue to the preceding word: "hello ," → "hello,".
                paragraphs[-1][-1] = paragraphs[-1][-1] + replacement
            else:
                # Paragraph starts with a punctuation command (rare —
                # e.g. "new paragraph comma foo"). Prepend standalone.
                paragraphs[-1].append(replacement)
            i += n
            continue

        paragraphs[-1].append(tokens[i])
        i += 1

    joined = [" ".join(p) for p in paragraphs if p]
    # Preserve the "always at least one element" invariant — an input
    # like "new paragraph" alone filters to nothing otherwise.
    return joined if joined else [""]


def _match_layout(tokens: list[str], i: int) -> int | None:
    """Return the phrase length if a layout command matches at `tokens[i]`,
    else None."""
    for phrase in _LAYOUT_PHRASES:
        phrase_toks = phrase.split()
        n = len(phrase_toks)
        if tokens[i : i + n] != phrase_toks:
            continue
        if not _is_command_context(tokens, i, n, phrase):
            continue
        return n
    return None


def _match_punct(tokens: list[str], i: int) -> tuple[str, int] | None:
    """Return `(replacement, phrase_length)` if a punctuation command
    matches at `tokens[i]`, else None."""
    for phrase, replacement in _ALL_COMMANDS:
        phrase_toks = phrase.split()
        n = len(phrase_toks)
        if tokens[i : i + n] != phrase_toks:
            continue
        if not _is_command_context(tokens, i, n, phrase):
            continue
        return replacement, n
    return None


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
