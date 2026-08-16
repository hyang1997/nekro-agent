"""Body cleanup for the Gmail plugin.

This is the failure-prone part. Marketing mail combines nested-table layout with
zero-width preheader padding, so naive tag-stripping yields hundreds of lines that look
empty but are not — burying the real content and burning context. Measured against the
live mailbox: the Garmin sign-in alert is one line of text paved over with U+200C.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "hao_gmail_plugin",
    Path(__file__).resolve().parent.parent / "plugins" / "workdir" / "gmail.py",
)
assert _SPEC and _SPEC.loader
gmail = importlib.util.module_from_spec(_SPEC)
sys.modules["hao_gmail_plugin"] = gmail
_SPEC.loader.exec_module(gmail)

ZWNJ = "‌"
NBSP = " "


class TestTidyLines:
    def test_plain_text_survives(self):
        assert gmail._tidy_lines("hello\nworld") == "hello\nworld"

    def test_zero_width_padding_is_removed(self):
        padded = "Sign-in from New Location\n" + (ZWNJ + " ") * 400 + "\nreal content"
        out = gmail._tidy_lines(padded)
        assert ZWNJ not in out
        assert out.splitlines()[0] == "Sign-in from New Location"
        assert "real content" in out
        assert len(out.splitlines()) <= 4, f"invisible padding survived:\n{out[:200]}"

    @pytest.mark.parametrize(
        "ch",
        ["​", "‌", "‍", "‎", "﻿", "­", "͏", "⁠"],
    )
    def test_each_invisible_char_is_stripped(self, ch: str):
        assert gmail._tidy_lines(f"a{ch}b") == "ab"

    def test_nbsp_becomes_space(self):
        assert gmail._tidy_lines(f"a{NBSP}b") == "a b"

    def test_runs_of_spaces_collapse(self):
        assert gmail._tidy_lines("a        b") == "a b"

    def test_blank_run_collapses_to_one(self):
        assert gmail._tidy_lines("a\n\n\n\n\nb") == "a\n\nb"

    def test_paragraph_break_is_preserved(self):
        assert gmail._tidy_lines("para one\n\npara two") == "para one\n\npara two"

    def test_whitespace_only_input(self):
        assert gmail._tidy_lines(f"  \n{ZWNJ}\n {NBSP} \n") == ""

    def test_empty_input(self):
        assert gmail._tidy_lines("") == ""

    def test_leading_and_trailing_blank_lines_trimmed(self):
        assert gmail._tidy_lines("\n\n  body  \n\n") == "body"


class TestHtmlToText:
    def test_tags_removed_text_kept(self):
        assert "Hello" in gmail._html_to_text("<p>Hello</p>")

    def test_script_and_style_dropped(self):
        out = gmail._html_to_text("<style>.a{color:red}</style><script>alert(1)</script><p>Body</p>")
        assert "color:red" not in out
        assert "alert" not in out
        assert "Body" in out

    def test_entities_unescaped(self):
        out = gmail._html_to_text("<p>a &amp; b &lt;c&gt; &quot;d&quot; &#39;e&#39;</p>")
        assert "&amp;" not in out
        assert "a & b <c> \"d\" 'e'" in out

    def test_br_becomes_newline(self):
        assert gmail._html_to_text("a<br>b").splitlines() == ["a", "b"]

    def test_block_tags_separate_paragraphs(self):
        assert gmail._html_to_text("<p>one</p><p>two</p>").splitlines() == ["one", "two"]

    def test_adjacent_blocks_do_not_run_together(self):
        """Without a break at </div>, 'Total' and '$42' would fuse into one token."""
        assert "Total$42" not in gmail._html_to_text("<div>Total</div><div>$42</div>")

    def test_table_email_does_not_explode_into_blank_lines(self):
        """Nested tables are the norm in marketing mail; stripping must not leave a screen of blanks."""
        html = "<table>" + "<tr><td>&nbsp;</td></tr>" * 300 + "<tr><td>Actual content</td></tr></table>"
        out = gmail._html_to_text(html)
        assert "Actual content" in out
        assert len(out.splitlines()) <= 5, f"table layout produced {len(out.splitlines())} lines"

    def test_preheader_padding_email(self):
        """Reproduces the observed Garmin message: one line of text plus a field of U+200C."""
        html = f"<div>Sign-in from New Location</div><div>{(ZWNJ + ' ') * 500}</div><div>Toronto, CA</div>"
        out = gmail._html_to_text(html)
        assert ZWNJ not in out
        assert "Sign-in from New Location" in out
        assert "Toronto, CA" in out
        assert len(out) < 200, f"still {len(out)} characters after cleanup"


class TestDecodeHeader:
    def test_plain_header(self):
        assert gmail._decode_header("Hello World") == "Hello World"

    def test_none_header(self):
        assert gmail._decode_header(None) == ""

    def test_rfc2047_base64(self):
        # =?UTF-8?B?5L2g5aW9?= decodes to a two-character CJK string
        assert gmail._decode_header("=?UTF-8?B?5L2g5aW9?=") == "你好"

    def test_rfc2047_quoted_printable(self):
        assert "Fwd" in gmail._decode_header("=?utf-8?Q?Fwd=3A_test?=")

    def test_mixed_encoded_and_plain(self):
        out = gmail._decode_header("=?UTF-8?B?5L2g5aW9?= world")
        assert "你好" in out
        assert "world" in out


class TestParseLabels:
    def test_no_labels(self):
        assert gmail._parse_labels("* 1 FETCH (UID 5)") == ""

    def test_single_label(self):
        assert "Important" in gmail._parse_labels(r"* 1 FETCH (X-GM-LABELS (\\Important) UID 5)")

    def test_multiple_labels(self):
        out = gmail._parse_labels(r'* 1 FETCH (X-GM-LABELS (\\Important "Receipts") UID 5)')
        assert "Important" in out
        assert "Receipts" in out
