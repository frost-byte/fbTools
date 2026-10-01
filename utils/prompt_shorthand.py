"""Pure text-expansion logic for the S1/V1/A1 prompt shorthand convention.

No ComfyUI dependencies -- testable in isolation. See
docs/openshot-bridge-prompt-shorthand.md for the user-facing convention this
implements: a user writes S1/S2/.../V1/V2/A1/A2 in a freeform prompt, and this
expands them into MiniMax H3's native <Subject 1>/<Video 1>/<Audio 1>
reference-label format before the prompt reaches the model.
"""
import re

_SHORTHAND_PATTERNS = (
    (re.compile(r"\bS(\d+)\b", re.IGNORECASE), r"<Subject \1>"),
    (re.compile(r"\bV(\d+)\b", re.IGNORECASE), r"<Video \1>"),
    (re.compile(r"\bA(\d+)\b", re.IGNORECASE), r"<Audio \1>"),
)


def expand_prompt_shorthand(text: str) -> str:
    """Expand S1/V1/A1-style shorthand into MiniMax H3's native
    <Subject 1>/<Video 1>/<Audio 1> reference-label format.

    Word-bounded and case-insensitive, so "s1"/"S1" both expand but a
    stray "xS1" or "S1x" token does not. Text already using the full
    <Subject N> form has no bare S<digits>/V<digits>/A<digits> tokens to
    match, so passing already-expanded text through is a safe no-op.
    """
    text = str(text or "")
    for pattern, replacement in _SHORTHAND_PATTERNS:
        text = pattern.sub(replacement, text)
    return text
