import hashlib
import re

import chardet


def calculate_file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


def detect_encoding(file_bytes: bytes) -> str:
    """旧版兼容：仅返回预测编码名。"""
    result = chardet.detect(file_bytes)
    encoding = result["encoding"]
    if not encoding or result["confidence"] < 0.5:
        return "utf-8-sig"
    if encoding.lower() == "gb2312":
        return "gb18030"
    return encoding


def decode_subtitle_bytes(file_bytes: bytes) -> tuple[str, str, bool]:
    """
    解码字幕字节流。
    返回: (解码后的文本, 实际使用的编码, 是否使用了替换字符进行容错解码)
    """
    # 1. 优先尝试 chardet 检测出的编码（如果是高置信度）
    result = chardet.detect(file_bytes)
    detected_enc = result.get("encoding")
    confidence = result.get("confidence") or 0

    if detected_enc:
        enc = detected_enc.lower()
        if enc == "gb2312":
            enc = "gb18030"
        if confidence >= 0.5:
            try:
                content = file_bytes.decode(enc)
                if "\ufffd" not in content:
                    return content, enc, False
            except Exception:
                pass

    # 2. 依次尝试常用的 fallback 编码（严格解码模式）
    fallbacks = [
        "utf-8-sig",
        "utf-8",
        "gb18030",
        "big5",
        "cp950",
        "shift_jis",
        "cp932",
        "utf-16",
    ]

    for enc in fallbacks:
        try:
            # 如果 chardet 已经试过了，就不再重复试
            if detected_enc and enc == detected_enc.lower():
                continue
            content = file_bytes.decode(enc)
            # utf-16 有时会把二进制误识别，这里做个简单的启发式检查：
            # 字幕文件通常包含一些标点符号或数字
            if enc == "utf-16" and len(content) > 10:
                if not any(c in content for c in "0123456789:->"):
                    continue
            return content, enc, False
        except Exception:
            continue

    # 3. 最后无奈之举：使用 utf-8-sig 容错解码
    return file_bytes.decode("utf-8-sig", errors="replace"), "utf-8-sig", True


def clean_subtitle_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\{[^}]+\}", "", text)
    return text.strip()


def parse_ass_content(content: str) -> list[dict]:
    parsed_data = []
    for line in content.split("\n"):
        if line.startswith("Dialogue:"):
            try:
                parts = line.split(",", 9)
                if len(parts) >= 10:
                    clean_text = parts[9].replace(r"\N", " ").replace(r"\n", " ")
                    clean_text = clean_subtitle_text(clean_text)
                    if clean_text:
                        parsed_data.append(
                            {
                                "start": parts[1].strip(),
                                "end": parts[2].strip(),
                                "text": clean_text,
                            }
                        )
            except Exception:
                continue
    return parsed_data


def parse_srt_content(content: str) -> list[dict]:
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    pattern_std = re.compile(
        r"\d+\s*\n"
        r"(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})"
        r"\s*\n([\s\S]*?)(?=\n\n|\n\d+\s*\n|\Z)",
        re.MULTILINE,
    )
    matches = pattern_std.findall(content)
    if matches:
        parsed_data = []
        for m in matches:
            clean_text = clean_subtitle_text(m[2]).replace("\n", " ")
            if clean_text:
                parsed_data.append(
                    {
                        "start": m[0].replace(",", "."),
                        "end": m[1].replace(",", "."),
                        "text": clean_text,
                    }
                )
        return parsed_data

    parsed_data = []
    pattern_fallback = re.compile(
        r"(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})(.*)"
    )
    current_entry = None
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        match = pattern_fallback.search(line)
        if match:
            if current_entry:
                parsed_data.append(current_entry)
            start, end, inline_text = match.groups()
            text_content = inline_text.strip()
            text_content = re.sub(r"\[.*?\]", "", text_content).strip()
            current_entry = {
                "start": start.replace(",", "."),
                "end": end.replace(",", "."),
                "text": text_content,
            }
        elif (
            current_entry
            and not line.isdigit()
            and not line.startswith("<")
            and not line.startswith("[")
        ):
            current_entry["text"] = (current_entry["text"] + " " + line).strip()
    if current_entry:
        parsed_data.append(current_entry)
    return parsed_data


def parse_vtt_content(content: str) -> list[dict]:
    return parse_srt_content(content.replace("WEBVTT", ""))


def parse_subtitle_content(content: str, ext: str) -> list[dict]:
    """按扩展名解析字幕内容。"""
    ext = ext.lower()
    if "[Script Info]" in content[:1000] and "Dialogue:" in content:
        return parse_ass_content(content)
    if ext in ("ass", "ssa"):
        return parse_ass_content(content)
    if ext == "vtt":
        return parse_vtt_content(content)
    if ext == "txt":
        return [
            {"start": "0", "end": "0", "text": line.strip()}
            for line in content.split("\n")
            if line.strip()
        ]
    return parse_srt_content(content)
