# chunk_slides.py
from __future__ import annotations
from pathlib import Path
import json
import re
from typing import List, Dict, Any

IN_PATH = Path("slides.jsonl")
OUT_PATH = Path("chunks.jsonl")

# ===== チューニングポイント =====
CHUNK_SIZE = 1000     # 目標サイズ（文字数）
CHUNK_OVERLAP = 180   # オーバーラップ（文字数）
MIN_SENT_LEN = 5      # あまりに短い断片は隣とまとめる
# ==============================

# 日本語・英語混在を想定したゆるい文分割
SENT_SPLIT_RE = re.compile(r"(?<=[。．！？!?])\s*(?=\S)|\n{2,}")

def normalize_text(s: str) -> str:
    # 連続スペースやタブを整理。箇条書きの改行は残す。
    s = s.replace("\u3000", " ")  # 全角スペース→半角
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def split_into_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    parts = SENT_SPLIT_RE.split(text)
    # さらに1行の中で改行が多い場合は軽く整える（見出し行など）
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # 箇条書き連続はそのまま、1行が極端に長ければ句読点で再分割
        if len(p) > CHUNK_SIZE * 1.8:
            tmp = re.split(r"(?<=[。．！？!?])", p)
            pbuf = ""
            for t in tmp:
                pbuf += t
                if len(pbuf) >= CHUNK_SIZE:
                    out.append(pbuf.strip())
                    pbuf = ""
            if pbuf:
                out.append(pbuf.strip())
        else:
            out.append(p)
    # 短すぎる文のマージ
    merged = []
    buf = ""
    for sent in out:
        if len(sent) < MIN_SENT_LEN:
            buf += (("\n" if buf else "") + sent)
            continue
        if not buf:
            merged.append(sent)
        else:
            if len(buf) < MIN_SENT_LEN:
                merged.append(buf + ("\n" + sent))
            else:
                merged.append(buf)
                merged.append(sent)
            buf = ""
    if buf:
        merged.append(buf)
    return merged

def pack_sentences(sentences: List[str], size: int, overlap: int) -> List[str]:
    """
    文リストを「目標サイズ size」に近づけつつ塊にまとめ、隣接チャンクに overlap 文字の重なりを持たせる。
    """
    if not sentences:
        return []
    # まずは貪欲にパッキング
    chunks: List[str] = []
    buf = ""
    for s in sentences:
        if not buf:
            buf = s
            continue
        if len(buf) + 1 + len(s) <= size:
            buf = f"{buf}\n{s}"
        else:
            chunks.append(buf)
            buf = s
    if buf:
        chunks.append(buf)

    # オーバーラップ付与：直前チャンク末尾 overlap 文字を次頭に重ねる
    if overlap > 0 and chunks:
        overlapped: List[str] = []
        prev_tail = ""
        for i, ch in enumerate(chunks):
            if i == 0:
                overlapped.append(ch)
            else:
                # 前の末尾を付け足す。ただし重複しすぎないようトリム
                tail = prev_tail[-overlap:]
                # 既に同内容で始まっていなければ重ねる
                if tail and not ch.startswith(tail):
                    ch2 = (tail + "\n" + ch).strip()
                else:
                    ch2 = ch
                overlapped.append(ch2)
            prev_tail = ch
        chunks = overlapped
    return chunks

def chunk_record(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    1スライドのレコード -> 複数チャンクのレコード群
    """
    text = rec.get("text", "") or ""
    sentences = split_into_sentences(text)
    chunks = pack_sentences(sentences, CHUNK_SIZE, CHUNK_OVERLAP)

    # char_start/end をオリジナル本文内の位置で追跡（おおよその位置でOK）
    # 簡易には順次スライスで積み上げ
    out: List[Dict[str, Any]] = []
    cursor = 0
    for idx, ch in enumerate(chunks):
        # 位置探索（同一テキスト繰り返し対策に find の開始位置を前進）
        pos = text.find(ch[:50], cursor)  # 先頭50文字でだいたいの位置を見つける
        if pos == -1:
            pos = cursor
        start = pos
        end = min(len(text), start + len(ch))
        cursor = end

        item = {
            "doc_name": rec.get("doc_name"),
            "path": rec.get("path"),
            "slide": rec.get("slide"),
            "chunk_index": idx,
            "char_start": start,
            "char_end": end,
            "text": ch.strip()
        }
        out.append(item)
    return out

def main():
    if not IN_PATH.exists():
        print(f"[ERR] not found: {IN_PATH.resolve()}")
        return

    count_in = 0
    count_out = 0
    with OUT_PATH.open("w", encoding="utf-8") as w, IN_PATH.open("r", encoding="utf-8") as r:
        for line in r:
            if not line.strip():
                continue
            rec = json.loads(line)
            count_in += 1
            chunks = chunk_record(rec)
            for ch in chunks:
                ch["char_count"] = len(ch["text"])
                w.write(json.dumps(ch, ensure_ascii=False) + "\n")
                count_out += 1

    print(f"✅ slides: {count_in} → chunks: {count_out}")
    # ざっくり統計
    try:
        import statistics as stats
        lengths = []
        with OUT_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                lengths.append(json.loads(line)["char_count"])
        if lengths:
            print(f"   平均文字数: {int(stats.mean(lengths))} / 中央: {int(stats.median(lengths))} / 最大: {max(lengths)}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
