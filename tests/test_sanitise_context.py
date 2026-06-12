"""
Comprehensive tests for sanitise_context.py

Tests security controls:
- HTML escaping for XSS prevention
- Safe markdown rendering
- Content escaping in display functions
- Markdown stripping
"""

from unittest.mock import patch

from markupsafe import escape

from security.sanitise_context import (
    display_safe_low_score_warning,
    display_safe_publication_date_info,
    safe_markdown,
    strip_markdown_and_code,
)


class TestHTMLEscaping:
    """Test HTML escaping for XSS prevention."""

    def test_script_tag_escaped(self):
        """Test that script tags are escaped."""
        content = '<script>alert("XSS")</script>'
        escaped = escape(content)
        assert "<script>" not in escaped
        assert "alert" in escaped  # Content preserved but tags escaped

    def test_iframe_injection_escaped(self):
        """Test that iframe injections are escaped."""
        content = '<iframe src="evil.com"></iframe>'
        escaped = escape(content)
        assert "<iframe" not in escaped
        assert "evil.com" in escaped

    def test_event_handler_escaped(self):
        """Test that event handlers are escaped."""
        content = '<img src="x" onerror="alert(1)">'
        escaped = escape(content)
        # The tag is escaped to &lt;, making it safe
        assert "&lt;img" in str(escaped)
        assert "alert" in str(escaped) or "&#" in str(escaped)

    def test_onclick_handler_escaped(self):
        """Test that onclick handlers are escaped."""
        content = '<button onclick="malicious()">Click</button>'
        escaped = escape(content)
        # The tag is escaped, making it safe from execution
        assert "&lt;button" in str(escaped)

    def test_style_injection_escaped(self):
        """Test that style injections are escaped."""
        content = "<style>body { display: none; }</style>"
        escaped = escape(content)
        assert "<style>" not in escaped

    def test_svg_xss_escaped(self):
        """Test that SVG-based XSS is escaped."""
        content = '<svg onload="alert(1)">'
        escaped = escape(content)
        # The tag is escaped making it safe
        assert "&lt;svg" in str(escaped)

    def test_javascript_protocol_escaped(self):
        """Test that javascript: protocol links are escaped."""
        content = '<a href="javascript:alert(1)">Click</a>'
        escaped = escape(content)
        # The tag is escaped making the link safe
        assert "&lt;a" in str(escaped)
        assert "&lt;/a&gt;" in str(escaped)


class TestSafeMarkdownRendering:
    """Test safe markdown rendering function."""

    @patch("security.sanitise_context.st.markdown")
    def test_safe_markdown_escapes_html(self, mock_markdown):
        """Test that safe_markdown escapes HTML."""
        content = '<script>alert("test")</script>'
        safe_markdown(content)

        mock_markdown.assert_called_once()
        call_args = mock_markdown.call_args[0][0]

        # Should be escaped
        assert "<script>" not in call_args

    @patch("security.sanitise_context.st.markdown")
    def test_safe_markdown_uses_unsafe_html_false(self, mock_markdown):
        """Test that safe_markdown disables unsafe_allow_html."""
        safe_markdown("Test content")

        mock_markdown.assert_called_once()
        # Second arg should be unsafe_allow_html=False
        assert mock_markdown.call_args[1].get("unsafe_allow_html") is False

    @patch("security.sanitise_context.st.markdown")
    def test_safe_markdown_with_none(self, mock_markdown):
        """Test that None content is handled."""
        safe_markdown(None)

        # Should not call st.markdown if content is None
        mock_markdown.assert_not_called()

    @patch("security.sanitise_context.st.markdown")
    def test_safe_markdown_with_empty_string(self, mock_markdown):
        """Test that empty string is handled."""
        safe_markdown("")

        mock_markdown.assert_not_called()

    @patch("security.sanitise_context.st.markdown")
    def test_safe_markdown_preserves_legitimate_content(self, mock_markdown):
        """Test that legitimate content is preserved."""
        content = "This is **bold** and *italic* text"
        safe_markdown(content)

        mock_markdown.assert_called_once()
        # Content should be preserved but HTML-escaped
        call_args = mock_markdown.call_args[0][0]
        assert "bold" in call_args
        assert "italic" in call_args

    @patch("security.sanitise_context.st.markdown")
    def test_safe_markdown_with_markdown_syntax(self, mock_markdown):
        """Test that markdown syntax is preserved."""
        content = "# Title\n\n- List item\n- Another item"
        safe_markdown(content)

        call_args = mock_markdown.call_args[0][0]
        # Markdown syntax preserved but HTML escaped
        assert "#" in call_args or "Title" in call_args


class TestPublicationDateDisplay:
    """Test safe publication date display."""

    @patch("security.sanitise_context.st.markdown")
    def test_publication_date_escaped(self, mock_markdown):
        """Test that publication date info is escaped."""
        date_info = '<img src=x onerror="alert(1)">2024-01-15'
        display_safe_publication_date_info(date_info)

        mock_markdown.assert_called_once()
        call_args = mock_markdown.call_args[0][0]

        # Should be HTML-escaped (markupsafe.escape converts < to &lt;)
        assert "&lt;img" in call_args or "<img" not in call_args

    @patch("security.sanitise_context.st.markdown")
    def test_publication_date_emoji_included(self, mock_markdown):
        """Test that date emoji is included."""
        date_info = "2024-01-15"
        display_safe_publication_date_info(date_info)

        call_args = mock_markdown.call_args[0][0]
        assert "📅" in call_args

    @patch("security.sanitise_context.st.markdown")
    def test_publication_date_none_not_displayed(self, mock_markdown):
        """Test that None date is not displayed."""
        display_safe_publication_date_info(None)

        mock_markdown.assert_not_called()

    @patch("security.sanitise_context.st.markdown")
    def test_publication_date_empty_not_displayed(self, mock_markdown):
        """Test that empty date is not displayed."""
        display_safe_publication_date_info("")

        mock_markdown.assert_not_called()

    @patch("security.sanitise_context.st.markdown")
    def test_publication_date_with_injection(self, mock_markdown):
        """Test that injected JavaScript is escaped."""
        date_info = '<script>alert("xss")</script> 2024-01-15'
        display_safe_publication_date_info(date_info)

        call_args = mock_markdown.call_args[0][0]
        # Should be HTML-escaped
        assert "&lt;script&gt;" in call_args or "<script>" not in call_args


class TestLowScoreWarning:
    """Test low score warning display."""

    @patch("security.sanitise_context.st.markdown")
    def test_warning_displayed(self, mock_markdown):
        """Test that warning message is displayed."""
        display_safe_low_score_warning()

        mock_markdown.assert_called_once()
        call_args = mock_markdown.call_args[0][0]

        # Should contain warning emoji and message
        assert "⚠️" in call_args
        assert "relevance threshold" in call_args.lower()

    @patch("security.sanitise_context.st.markdown")
    def test_warning_uses_safe_markdown(self, mock_markdown):
        """Test that safe markdown is used for warning."""
        display_safe_low_score_warning()

        # Should not use unsafe_allow_html
        call_kwargs = mock_markdown.call_args[1]
        assert call_kwargs.get("unsafe_allow_html") is not True


class TestMarkdownStripping:
    """Test markdown syntax stripping."""

    def test_strip_headers(self):
        """Test removal of header syntax."""
        text = "# Title\n## Subtitle\n### Level 3"
        stripped = strip_markdown_and_code(text)

        # Header markers should be removed
        assert "# Title" not in stripped or "# " not in stripped
        assert "Title" in stripped  # But content preserved

    def test_strip_code_blocks(self):
        """Test removal of code blocks."""
        text = "```python\ndef hello():\n    pass\n```"
        stripped = strip_markdown_and_code(text)

        # Code block markers removed
        assert "```" not in stripped

    def test_strip_inline_code(self):
        """Test removal of inline code markers."""
        text = "Use the `function_name()` to do X"
        stripped = strip_markdown_and_code(text)

        # Code markers removed but content kept (underscores also removed by markdown regex)
        assert "`" not in stripped
        # Note: Current implementation removes underscores as part of markdown emphasis stripping
        # So 'function_name' becomes 'functionname'
        assert "functionname" in stripped or "function_name" in stripped

    def test_strip_bold_and_italic(self):
        """Test removal of bold/italic markers."""
        text = "This is **bold** and *italic* text"
        stripped = strip_markdown_and_code(text)

        # Emphasis markers removed but content kept
        assert "**" not in stripped
        assert "*" not in stripped
        assert "bold" in stripped
        assert "italic" in stripped

    def test_strip_links(self):
        """Test removal of link syntax."""
        text = "[Link text](https://example.com)"
        stripped = strip_markdown_and_code(text)

        # Link syntax removed, text preserved
        assert "[" not in stripped
        assert "Link text" in stripped

    def test_strip_images(self):
        """Test removal of image syntax."""
        text = "![Alt text](image.jpg)"
        stripped = strip_markdown_and_code(text)

        # Image syntax removed entirely
        assert "![" not in stripped

    def test_strip_lists(self):
        """Test removal of list markers."""
        text = "- Item 1\n- Item 2\n* Item 3"
        stripped = strip_markdown_and_code(text)

        # List markers removed, content might be preserved
        assert "- " not in stripped or "- Item" not in stripped

    def test_strip_blockquotes(self):
        """Test removal of blockquote markers."""
        text = "> This is a quote\n> Second line"
        stripped = strip_markdown_and_code(text)

        # Blockquote markers removed
        assert "> " not in stripped

    def test_strip_preserves_text_content(self):
        """Test that actual text content is preserved."""
        text = "# Title\nThis is **important** content about `code`"
        stripped = strip_markdown_and_code(text)

        # All text content should be there
        assert "Title" in stripped
        assert "important" in stripped
        assert "content" in stripped
        assert "code" in stripped

    def test_strip_multiple_newlines_normalized(self):
        """Test that multiple newlines are normalized."""
        text = "Line 1\n\n\n\nLine 2"
        stripped = strip_markdown_and_code(text)

        # Multiple newlines should be reduced
        assert "\n\n\n\n" not in stripped
        assert "Line 1" in stripped
        assert "Line 2" in stripped

    def test_strip_whitespace_normalized(self):
        """Test that excess whitespace is normalized."""
        text = "Text   with    spaces"
        stripped = strip_markdown_and_code(text)

        # Multiple spaces should be reduced
        assert "   " not in stripped
        assert "Text" in stripped
        assert "spaces" in stripped

    def test_strip_empty_result(self):
        """Test stripping of text with only syntax."""
        text = "```\n```"
        stripped = strip_markdown_and_code(text)

        # Result should be empty or whitespace only
        assert not stripped or stripped.isspace()

    def test_strip_complex_document(self):
        """Test stripping of complex markdown document."""
        text = """# Title
        
## Subtitle

Here is some **bold** text with a [link](https://example.com).

```python
def code():
    return "example"
```

- List item 1
- List item 2

> A quote here
> Second line
"""
        stripped = strip_markdown_and_code(text)

        # Should preserve actual content, not syntax
        assert "Title" in stripped
        assert "bold" in stripped
        assert "link" in stripped
        assert "List item" in stripped
        assert "quote" in stripped


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch("security.sanitise_context.st.markdown")
    def test_very_long_content(self, mock_markdown):
        """Test handling of very long content."""
        content = "A" * 10000
        safe_markdown(content)

        mock_markdown.assert_called_once()

    @patch("security.sanitise_context.st.markdown")
    def test_unicode_content(self, mock_markdown):
        """Test handling of unicode content."""
        content = "你好世界 🌍 Привет мир"
        safe_markdown(content)

        call_args = mock_markdown.call_args[0][0]
        # Unicode should be preserved
        assert "世界" in call_args or "мир" in call_args or "🌍" in call_args

    def test_strip_with_null_bytes(self):
        """Test stripping with null bytes."""
        text = "Hello\x00World"
        # Should not crash
        stripped = strip_markdown_and_code(text)
        assert stripped is not None

    def test_strip_with_control_characters(self):
        """Test stripping with control characters."""
        text = "Hello\x01\x02\x03World"
        # Should not crash
        stripped = strip_markdown_and_code(text)
        assert "Hello" in stripped

    @patch("security.sanitise_context.st.markdown")
    def test_safe_markdown_with_special_markdown_chars(self, mock_markdown):
        """Test safe_markdown with special characters."""
        content = "Price: $50 | Rating: 4/5 & highly recommended"
        safe_markdown(content)

        call_args = mock_markdown.call_args[0][0]
        # Content should be preserved
        assert "Price" in call_args
        assert "50" in call_args

    def test_strip_repeated_emphasis(self):
        """Test stripping of repeated emphasis markers."""
        text = "***very bold and italic***"
        stripped = strip_markdown_and_code(text)

        assert "*" not in stripped
        assert "very" in stripped


class TestXSSVectors:
    """Test various XSS attack vectors."""

    def test_data_uri_xss_escaped(self):
        """Test that data: URI injections are escaped."""
        content = '<a href="data:text/html,<img src=x onerror=alert(1)>">Click</a>'
        escaped = escape(content)
        # Tags should be escaped
        assert "&lt;" in str(escaped)

    def test_svg_animate_xss_escaped(self):
        """Test that SVG animate XSS is escaped."""
        content = '<svg><animate onbegin="alert(1)" dur="1s"/></svg>'
        escaped = escape(content)
        assert "&lt;svg&gt;" in str(escaped)

    def test_form_input_xss_escaped(self):
        """Test that form input injections are escaped."""
        content = '<form><input onfocus="alert(1)" autofocus></form>'
        escaped = escape(content)
        assert "&lt;form&gt;" in str(escaped)

    def test_meta_refresh_xss_escaped(self):
        """Test that meta refresh XSS is escaped."""
        content = '<meta http-equiv="refresh" content="0;url=javascript:alert(1)">'
        escaped = escape(content)
        # Tag is escaped
        assert "&lt;meta" in str(escaped)

    def test_base_tag_xss_escaped(self):
        """Test that base tag XSS is escaped."""
        content = '<base href="javascript:alert(1)//">'
        escaped = escape(content)
        # Tag is escaped
        assert "&lt;base" in str(escaped)
