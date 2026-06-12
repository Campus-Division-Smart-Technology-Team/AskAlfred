import re
from typing import Optional

import streamlit as st
from markupsafe import escape

CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`([^`]*)`")
MARKDOWN_REPLACEMENTS = [
    (re.compile(r"!\[.*?\]\(.*?\)"), ""),  # images
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),  # links → text
    (re.compile(r"^#{1,6}\s*", re.MULTILINE), ""),  # headers
    (re.compile(r"^\s*>\s?", re.MULTILINE), ""),  # blockquotes
    (re.compile(r"[*_~]{1,3}"), ""),  # emphasis
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

    # 4) Normalise whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def safe_markdown(content: Optional[str]) -> None:
    """
    Safely render markdown content with HTML escaping.

    Prevents XSS attacks by escaping HTML special characters before rendering.
    This function should be used for all user-generated or untrusted content.

    Args:
        content: The content to render safely

    Example:
        instead of: st.markdown(user_input)
        use:        safe_markdown(user_input)
    """
    if not content:
        return

    # Escape HTML special characters to prevent XSS
    safe_content = escape(content)
    st.markdown(safe_content, unsafe_allow_html=False)


def display_safe_publication_date_info(publication_date_info: Optional[str]) -> None:
    """
    Safely display publication date information.

    Escapes HTML to prevent injection attacks instead of using unsafe_allow_html.

    Args:
        publication_date_info: Publication date information to display
    """
    if publication_date_info:
        # Escape the content and render as plain text with markdown formatting
        safe_content = escape(publication_date_info)
        st.markdown(f"📅 {safe_content}")


def display_safe_low_score_warning() -> None:
    """
    Safely display low score warning.

    Uses plain markdown instead of HTML to prevent injection attacks.
    """
    st.markdown("⚠️ **Results below relevance threshold**")
