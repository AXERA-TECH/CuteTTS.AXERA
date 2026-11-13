#pragma once
// Minimal SentencePiece unigram encoder for the C++ demo.
// The model is pre-converted on the host to assets/tokenizer.bin by
// python/export_assets.py, so the board needs no libsentencepiece.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace cutetts {

class Tokenizer {
public:
    bool load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) {
            std::fprintf(stderr, "cannot open tokenizer %s\n", path.c_str());
            return false;
        }
        char magic[4];
        if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "CTTK", 4)) {
            std::fclose(f);
            return false;
        }
        uint32_t n = 0;
        std::fread(&n, 4, 1, f);
        int32_t unk = 0;
        std::fread(&unk, 4, 1, f);
        unk_id_ = unk;
        uint32_t ns = 0;
        std::fread(&ns, 4, 1, f);
        for (uint32_t i = 0; i < ns; ++i) {
            uint16_t len = 0;
            std::fread(&len, 2, 1, f);
            std::string s(len, '\0');
            std::fread(&s[0], 1, len, f);
            int32_t id = 0;
            std::fread(&id, 4, 1, f);
            specials_.emplace_back(s, id);
            if (s.size() > max_special_) max_special_ = s.size();
        }
        pieces_.resize(n);
        for (uint32_t i = 0; i < n; ++i) {
            uint16_t len = 0;
            std::fread(&len, 2, 1, f);
            std::string s(len, '\0');
            std::fread(&s[0], 1, len, f);
            float score = 0.f;
            std::fread(&score, 4, 1, f);
            pieces_[i] = {s, score};
            table_[s] = {static_cast<int>(i), score};
            if (s.size() > max_piece_) max_piece_ = s.size();
        }
        std::fclose(f);
        return true;
    }

    int unk_id() const { return unk_id_; }

    // HuggingFace/tokenizer.py compatible: split on added tokens, strip each text
    // chunk, then run the unigram Viterbi encoder on "▁"-prefixed text.
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> out;
        std::string buf;
        size_t i = 0;
        while (i < text.size()) {
            int hit = -1;
            size_t hit_len = 0;
            for (const auto& kv : specials_) {
                if (text.compare(i, kv.first.size(), kv.first) == 0) {
                    if ((int)kv.first.size() > hit || hit < 0) { hit = kv.second; hit_len = kv.first.size(); }
                }
            }
            if (hit >= 0) {
                encode_chunk(buf, out);
                out.push_back(hit);
                buf.clear();
                i += hit_len;
            } else {
                buf.push_back(text[i++]);
            }
        }
        encode_chunk(buf, out);
        return out;
    }

private:
    static void append_utf8(std::string& s, uint32_t cp) {
        if (cp < 0x80) s.push_back((char)cp);
        else if (cp < 0x800) { s.push_back((char)(0xC0 | (cp >> 6))); s.push_back((char)(0x80 | (cp & 0x3F))); }
        else if (cp < 0x10000) { s.push_back((char)(0xE0 | (cp >> 12))); s.push_back((char)(0x80 | ((cp >> 6) & 0x3F))); s.push_back((char)(0x80 | (cp & 0x3F))); }
        else { s.push_back((char)(0xF0 | (cp >> 18))); s.push_back((char)(0x80 | ((cp >> 12) & 0x3F))); s.push_back((char)(0x80 | ((cp >> 6) & 0x3F))); s.push_back((char)(0x80 | (cp & 0x3F))); }
    }
    static uint32_t next_cp(const std::string& s, size_t i, size_t& n) {
        const unsigned char c = s[i];
        if (c < 0x80) { n = 1; return c; }
        if ((c >> 5) == 0x6) { n = 2; return ((c & 0x1F) << 6) | (s[i + 1] & 0x3F); }
        if ((c >> 4) == 0xE) { n = 3; return ((c & 0x0F) << 12) | ((s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F); }
        n = 4;
        return ((c & 0x07) << 18) | ((s[i + 1] & 0x3F) << 12) | ((s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F);
    }

    void encode_chunk(const std::string& raw, std::vector<int>& out) const {
        // strip ASCII whitespace (HF tokenizer does .strip() on the chunk)
        size_t b = 0, e = raw.size();
        while (b < e && (unsigned char)raw[b] <= 0x20) ++b;
        while (e > b && (unsigned char)raw[e - 1] <= 0x20) --e;
        if (b >= e) return;
        // NFKC-ish: full-width ASCII -> ASCII, ideographic space -> space
        std::string norm;
        for (size_t i = b; i < e;) {
            size_t n = 0;
            uint32_t cp = next_cp(raw, i, n);
            i += n;
            if (cp >= 0xFF01 && cp <= 0xFF5E) cp -= 0xFEE0;
            else if (cp == 0x3000) cp = ' ';
            if (cp == ' ' || cp == '\n' || cp == '\t' || cp == '\r') {
                if (!norm.empty() && norm.back() == ' ') continue;  // collapse whitespace
                cp = ' ';
            }
            append_utf8(norm, cp);
        }
        std::string s;
        for (char c : norm) {
            if (c == ' ') s += "\xE2\x96\x81";
            else s.push_back(c);
        }
        if (s.compare(0, 3, "\xE2\x96\x81") != 0) s = "\xE2\x96\x81" + s;

        // SentencePiece BPE: split into UTF-8 symbols, then greedily merge the
        // adjacent pair whose concatenation has the best (highest) piece score.
        std::vector<std::string> syms;
        for (size_t i = 0; i < s.size();) {
            size_t n = 0;
            next_cp(s, i, n);
            syms.push_back(s.substr(i, n));
            i += n;
        }
        std::vector<char> alive(syms.size(), 1);
        while (true) {
            float best = -1e30f;
            int bi = -1;
            for (size_t i = 0; i + 1 < syms.size(); ++i) {
                if (!alive[i]) continue;
                size_t j = i + 1;
                while (j < syms.size() && !alive[j]) ++j;
                if (j >= syms.size()) break;
                auto it = table_.find(syms[i] + syms[j]);
                if (it == table_.end()) continue;
                if (it->second.second > best) {
                    best = it->second.second;
                    bi = static_cast<int>(i);
                }
            }
            if (bi < 0) break;
            size_t j = bi + 1;
            while (j < syms.size() && !alive[j]) ++j;
            syms[bi] += syms[j];
            alive[j] = 0;
        }
        for (size_t i = 0; i < syms.size(); ++i) {
            if (!alive[i]) continue;
            auto it = table_.find(syms[i]);
            out.push_back(it == table_.end() ? unk_id_ : it->second.first);
        }
    }

    std::vector<std::pair<std::string, float>> pieces_;
    std::vector<std::pair<std::string, int>> specials_;
    std::unordered_map<std::string, std::pair<int, float>> table_;
    int unk_id_ = 0;
    size_t max_piece_ = 16;
    size_t max_special_ = 16;
};

}  // namespace cutetts
