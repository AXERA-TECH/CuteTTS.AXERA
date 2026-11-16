"""长文本按句切分（单次请求上限约 20.3 s 语音）。

规则与 C++ 实现（`cpp/src/text_split.hpp`）一致：
  1. 句末标点切分：。！？；…! ? ; 与换行
  2. 单句仍超长 → 退到 ，、, 处切 → 再不行按字符硬切
  3. 贪心合并到 max_seconds（默认 18 s，给 20.3 s 硬上限留余量）
"""

from __future__ import annotations

import re

_SENT_END = set("。！？；…!?;.\n")
_SUB_END = set("，、,")
# 句末标点后可能跟的收尾符号，并入前一句
_TRAILING = set("\"'”』 」)）]】")


def char_seconds(ch: str) -> float:
    """单个字符的估算口语时长（秒）。"""
    o = ord(ch)
    if 0x4E00 <= o <= 0x9FFF:      # 汉字
        return 0.22
    if 0x3040 <= o <= 0x30FF:      # 假名
        return 0.20
    if ch.isascii() and (ch.isalnum()):
        return 0.055
    if ch in " ,-":
        return 0.02
    return 0.0


def estimate_seconds(text: str) -> float:
    return sum(char_seconds(c) for c in text)


def _split_keep(text: str, marks: set) -> list[str]:
    """在指定标点后切分，标点留在前一段。"""
    out, buf = [], []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        buf.append(ch)
        i += 1
        if ch in marks:
            while i < n and (text[i] in _TRAILING or text[i] in " \t"):
                buf.append(text[i])
                i += 1
            seg = "".join(buf)
            if estimate_seconds(seg) > 0:
                out.append(seg)
            buf = []
    if buf:
        seg = "".join(buf)
        if estimate_seconds(seg) > 0:
            out.append(seg)
    return out


def _hard_split(text: str, max_seconds: float) -> list[str]:
    out, buf, t = [], [], 0.0
    for ch in text:
        d = char_seconds(ch)
        if t + d > max_seconds and buf:
            out.append("".join(buf))
            buf, t = [], 0.0
        buf.append(ch)
        t += d
    if buf:
        out.append("".join(buf))
    return out


def split_text(text: str, max_seconds: float = 18.0) -> list[str]:
    """把长文本切成若干段，每段估算时长 ≤ max_seconds（尽量按句边界）。"""
    if max_seconds <= 0:
        max_seconds = 18.0
    hard = max_seconds * 1.02

    pieces: list[str] = []
    for sent in _split_keep(text, _SENT_END):
        if estimate_seconds(sent) <= hard:
            pieces.append(sent)
            continue
        for sub in _split_keep(sent, _SUB_END):
            if estimate_seconds(sub) <= hard:
                pieces.append(sub)
            else:
                pieces.extend(_hard_split(sub, max_seconds))
    if not pieces:
        return [text]

    merged: list[str] = []
    cur = ""
    cur_t = 0.0
    for p in pieces:
        pt = estimate_seconds(p)
        if not cur:
            cur, cur_t = p, pt
        elif cur_t + pt <= max_seconds:
            cur += p
            cur_t += pt
        else:
            merged.append(cur)
            cur, cur_t = p, pt
    if cur:
        merged.append(cur)
    return merged or [text]
