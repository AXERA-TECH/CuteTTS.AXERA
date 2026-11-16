// UTF-8 aware sentence splitter for long-text TTS requests.
//
// 目的：单次请求的生成上限是 254 latent（≈20.3 s 语音，KV 384 − prompt 256 − 1 边界），
// 超过会被静默截断。长文本在 host 侧按句切开、逐句请求即可（音色缓存命中后每句 TTFA ≈0.5 s）。
//
// 规则：
//   1. 句末标点切分：。！？；…! ? ; 与换行
//   2. 单句超长：退到 ，、, 处切；再不行按字符硬切（不切断 UTF-8）
//   3. 贪心合并到 max_seconds（默认 18 s，给 20.3 s 上限留余量）
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace cutetts {

struct U8Char {
    size_t off;   // 字节偏移
    unsigned cp;  // Unicode 码点
    size_t len;   // 字节数
};

static inline std::vector<U8Char> u8_decode(const std::string& s) {
    std::vector<U8Char> out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        unsigned cp = c;
        size_t len = 1;
        if (c >= 0xF0) { cp = c & 0x07u; len = 4; }
        else if (c >= 0xE0) { cp = c & 0x0Fu; len = 3; }
        else if (c >= 0xC0) { cp = c & 0x1Fu; len = 2; }
        if (i + len > s.size()) len = 1, cp = c;
        for (size_t k = 1; k < len; ++k)
            cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3Fu);
        out.push_back({i, cp, len});
        i += len;
    }
    return out;
}

static inline double char_seconds(unsigned cp) {
    if (cp >= 0x4E00 && cp <= 0x9FFF) return 0.22;   // 汉字
    if (cp >= 0x3040 && cp <= 0x30FF) return 0.20;   // 假名
    if ((cp >= '0' && cp <= '9') || (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z'))
        return 0.055;                                 // 拉丁字母/数字
    if (cp == ' ' || cp == ',' || cp == '-') return 0.02;
    return 0.0;
}

static inline double estimate_seconds(const std::string& s) {
    double t = 0.0;
    for (const auto& ch : u8_decode(s)) t += char_seconds(ch.cp);
    return t;
}

static inline bool is_sentence_end(unsigned cp) {
    return cp == 0x3002 /*。*/ || cp == 0xFF01 /*！*/ || cp == 0xFF1F /*？*/ ||
           cp == 0xFF1B /*；*/ || cp == 0x2026 /*…*/ || cp == '.' || cp == '!' ||
           cp == '?' || cp == ';' || cp == '\n';
}

static inline bool is_sub_end(unsigned cp) {
    return cp == 0xFF0C /*，*/ || cp == 0x3001 /*、*/ || cp == ',';
}

// 在「句末标点」处切
static inline std::vector<std::string> split_sentences(const std::string& s) {
    const auto cs = u8_decode(s);
    std::vector<std::string> out;
    size_t start = 0;
    for (size_t i = 0; i < cs.size(); ++i) {
        if (!is_sentence_end(cs[i].cp)) continue;
        size_t cut = cs[i].off + cs[i].len;
        // 把紧跟的引号/空格并入本句
        size_t j = i + 1;
        while (j < cs.size() && (cs[j].cp == '"' || cs[j].cp == '\'' || cs[j].cp == 0x201D ||
                                  cs[j].cp == 0x300D || cs[j].cp == ' ' || cs[j].cp == '\t')) {
            cut = cs[j].off + cs[j].len;
            ++j;
        }
        std::string seg = s.substr(start, cut - start);
        if (estimate_seconds(seg) > 0.0) out.push_back(seg);
        start = cut;
        i = j - 1;
    }
    if (start < s.size()) {
        std::string tail = s.substr(start);
        if (estimate_seconds(tail) > 0.0) out.push_back(tail);
    }
    return out;
}

// 在次级标点处切
static inline std::vector<std::string> split_sub(const std::string& s) {
    const auto cs = u8_decode(s);
    std::vector<std::string> out;
    size_t start = 0;
    for (size_t i = 0; i < cs.size(); ++i) {
        if (!is_sub_end(cs[i].cp)) continue;
        size_t cut = cs[i].off + cs[i].len;
        std::string seg = s.substr(start, cut - start);
        if (estimate_seconds(seg) > 0.0) out.push_back(seg);
        start = cut;
    }
    if (start < s.size()) {
        std::string tail = s.substr(start);
        if (estimate_seconds(tail) > 0.0) out.push_back(tail);
    }
    return out;
}

// 按估算时长硬切（只在字符边界切）
static inline std::vector<std::string> hard_split(const std::string& s, double max_seconds) {
    const auto cs = u8_decode(s);
    std::vector<std::string> out;
    size_t b = 0;           // 当前片段起始（字符下标）
    double t = 0.0;
    for (size_t i = 0; i < cs.size(); ++i) {
        const double dt = char_seconds(cs[i].cp);
        if (t + dt > max_seconds && i > b) {
            out.push_back(s.substr(cs[b].off, cs[i].off - cs[b].off));
            b = i;
            t = 0.0;
        }
        t += dt;
    }
    if (b < cs.size())
        out.push_back(s.substr(cs[b].off));
    return out;
}

static inline std::vector<std::string> split_text(const std::string& text,
                                                 double max_seconds = 18.0) {
    if (max_seconds <= 0) max_seconds = 18.0;
    const double hard = max_seconds * 1.02;

    std::vector<std::string> pieces;
    for (const auto& s : split_sentences(text)) {
        if (estimate_seconds(s) <= hard) { pieces.push_back(s); continue; }
        for (const auto& p : split_sub(s)) {
            if (estimate_seconds(p) <= hard) { pieces.push_back(p); continue; }
            for (auto& q : hard_split(p, max_seconds)) pieces.push_back(q);
        }
    }
    if (pieces.empty()) return {text};

    // 贪心合并：尽量把多个短句凑到接近上限的一段
    std::vector<std::string> merged;
    std::string cur;
    for (const auto& s : pieces) {
        if (cur.empty()) { cur = s; continue; }
        if (estimate_seconds(cur) + estimate_seconds(s) <= max_seconds) cur += s;
        else { merged.push_back(cur); cur = s; }
    }
    if (!cur.empty()) merged.push_back(cur);
    return merged.empty() ? std::vector<std::string>{text} : merged;
}

}  // namespace cutetts
