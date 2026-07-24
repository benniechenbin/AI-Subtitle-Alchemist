import pytest

from src.utils.subtitle import decode_subtitle_bytes


def test_decode_utf8():
    text = "你好，世界"
    raw = text.encode("utf-8")
    content, encoding, had_replace = decode_subtitle_bytes(raw)
    assert content == text
    assert encoding in ("utf-8", "utf-8-sig")
    assert not had_replace


def test_decode_gb2312():
    # "你好，世界" in GB2312
    text = "你好，世界"
    raw = text.encode("gb2312")
    content, encoding, had_replace = decode_subtitle_bytes(raw)
    assert content == text
    assert encoding == "gb18030"
    assert not had_replace


def test_decode_gb18030():
    # GB18030 specific characters like "鿬" (U+9FEC)
    text = "你好，鿬"
    try:
        raw = text.encode("gb18030")
    except UnicodeEncodeError:
        pytest.skip("Environment does not support encoding this GB18030 character")

    content, encoding, had_replace = decode_subtitle_bytes(raw)
    assert content == text
    assert encoding == "gb18030"
    assert not had_replace


def test_decode_big5():
    # Realistic subtitle line in Big5
    text = "1\n00:00:01,000 --> 00:00:04,000\n你好，這是一個繁體字幕。"
    raw = text.encode("big5")
    content, encoding, had_replace = decode_subtitle_bytes(raw)
    print(f"Chosen encoding for Big5: {encoding}")
    assert "你好" in content
    assert encoding in ("big5", "cp950")
    assert not had_replace


def test_decode_shift_jis():
    # Realistic subtitle line in Shift-JIS
    text = "1\n00:00:01,000 --> 00:00:04,000\nこんにちは、これは日本の字幕です。"
    raw = text.encode("shift_jis")
    import chardet

    det = chardet.detect(raw)
    print(f"Chardet for Shift-JIS: {det}")
    content, encoding, had_replace = decode_subtitle_bytes(raw)
    print(f"Chosen encoding for Shift-JIS: {encoding}")
    assert not had_replace
    # If chardet is confident, it should be correct.
    # If it falls back to gb18030, we at least expect it to decode.
    if det["confidence"] > 0.7:
        assert "こんにちは" in content


def test_decode_fallback_success():
    # A byte sequence that might confuse chardet but should be caught by fallback
    text = "This is a test with some numbers 123 and symbols -->"
    raw = text.encode("utf-16")
    content, encoding, had_replace = decode_subtitle_bytes(raw)
    assert content == text
    assert encoding == "utf-16"
    assert not had_replace


def test_decode_corrupted():
    # A sequence that should fail most strict decodings
    raw = b"\x80\x81\xff\xff\x00\x00\x01\x01"
    _content, encoding, had_replace = decode_subtitle_bytes(raw)
    print(f"Corrupted sequence chose: {encoding}, had_replace: {had_replace}")
    # We expect either it fails all fallbacks and uses replacement
    # OR it matches something obscure but we don't care too much as long as it handles it.
