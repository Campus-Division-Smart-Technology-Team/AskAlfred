import re

CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`([^`]*)`")
MARKDOWN_REPLACEMENTS = [
    (re.compile(r"!\[.*?\]\(.*?\)"), ""),      # images
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),  # links → text
    (re.compile(r"^#{1,6}\s*", re.MULTILINE), ""),  # headers
    (re.compile(r"^\s*>\s?", re.MULTILINE), ""),    # blockquotes
    (re.compile(r"[*_~]{1,3}"), ""),            # emphasis
    (re.compile(r"^\s*[-+*]\s+", re.MULTILINE), ""),  # lists
]


def strip_markdown_and_code(text: str) -> str:
    if not text:
        return ""

    # 1) Remove fenced code blocks entirely
    text = CODE_BLOCK_RE.sub(" ", text)

    # 2) Remove inline code ticks but keep content
    text = INLINE_CODE_RE.sub(r"\1", text)

    # 3) Strip common markdown syntax
    for pattern, repl in MARKDOWN_REPLACEMENTS:
        text = pattern.sub(repl, text)

    # 4) Normalisse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()
