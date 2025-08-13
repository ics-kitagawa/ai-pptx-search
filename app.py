# app.py
from __future__ import annotations
from pathlib import Path
import os, textwrap, json, re
from typing import List, Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from openai import OpenAI

import re

# ====== 基本設定 ======
DB_DIR      = Path("vectordb")
COLLECTION  = "pptx_slides"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = os.getenv("CHAT_MODEL", "gpt-4o-mini")
TOP_K       = 15          # まずDBから拾う件数
TOP_K_LLM   = 6           # LLMに渡す件数（短く絞る）
CHUNKS_PATH = Path("chunks.jsonl")  # docフィルタ候補の取得に使う
# =====================

load_dotenv()

# ====== 便利: 文中ハイライト ======
def highlight(text: str, query: str) -> str:
    # ざっくりキーワードを抽出して強調（Embeddingは意味検索なので“おまけ”の可視化）
    words = [w for w in re.split(r"[^\w\u3040-\u30ff\u3400-\u9fff]+", query) if w]
    pat = re.compile("|".join(map(re.escape, sorted(set(words), key=len, reverse=True))), re.IGNORECASE)
    return pat.sub(lambda m: f"<mark>{m.group(0)}</mark>", text) if words else text

# 既存の highlight はそのまま使う（表では使わない）
def looks_like_markdown_table(text: str) -> bool:
    """
    ざっくり：| header | の行 と | --- | の区切り行があればテーブルと判断
    """
    if not text or "|" not in text:
        return False
    has_row = re.search(r"^\s*\|.*\|\s*$", text, flags=re.MULTILINE)
    has_sep = re.search(r"^\s*\|\s*:?-{3,}\s*(\|\s*:?-{3,}\s*)+\|\s*$", text, flags=re.MULTILINE)
    return bool(has_row and has_sep)

def render_block_with_markdown_and_highlight(block: str, query: str):
    block = block.strip()
    if not block:
        return
    if looks_like_markdown_table(block):
        # 表は素のMarkdownとして描画（テーブルが崩れない）
        st.markdown(block, unsafe_allow_html=False)
    else:
        # それ以外はハイライト（<mark>）OK
        html = highlight(block, query)  # ← ここで改行置換はしない！
        st.markdown(html, unsafe_allow_html=True)

# ====== 再ランク（Cross-Encoder） ======
_reranker = None
def get_reranker():
    global _reranker
    if _reranker is None:
        # 軽くて速い再ランク用（ローカル）
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")  # 自動DL
    return _reranker

def rerank(query: str, hits: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    if not hits:
        return hits
    model = get_reranker()
    pairs = [[query, h["text"]] for h in hits]
    scores = model.predict(pairs).tolist()
    for h, s in zip(hits, scores):
        h["rerank"] = float(s)
    hits.sort(key=lambda x: x.get("rerank", 0.0), reverse=True)
    return hits[:top_n]

# ====== ベクトル検索 ======
@st.cache_resource
def get_clients():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY が未設定です。.env か環境変数で設定してください。")
    oa = OpenAI()
    chroma = chromadb.PersistentClient(path=str(DB_DIR), settings=Settings(allow_reset=False))
    try:
        coll = chroma.get_collection(COLLECTION)
    except Exception:
        raise RuntimeError("Chroma コレクションが見つかりません。embed_chunks.py を先に実行してください。")
    return oa, coll

def embed_query(oa: OpenAI, text: str) -> List[float]:
    return oa.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding

def retrieve(oa: OpenAI, coll, query: str, k: int, doc_filter: Optional[List[str]]):
    qvec = embed_query(oa, query)
    where = {"doc_name": {"$in": doc_filter}} if doc_filter else None
    res = coll.query(query_embeddings=[qvec], n_results=k, where=where,
                     include=["documents", "metadatas", "distances"])
    docs   = res["documents"][0]
    metas  = res["metadatas"][0]
    dists  = res.get("distances", [[None]*len(docs)])[0]
    items = []
    for d, m, dist in zip(docs, metas, dists):
        if not d:
            continue
        items.append({"text": d, "meta": m, "score": 1.0 - (dist or 0)})
    return items

def build_messages(query: str, hits: List[Dict[str, Any]]) -> list:
    blocks = []
    for idx, h in enumerate(hits, start=1):
        m = h["meta"]
        head = f"[{idx}] {m.get('doc_name')} / slide {m.get('slide')} / chunk {m.get('chunk_index')}"
        body = h["text"].strip()
        blocks.append(head + "\n" + body)
    context = "\n\n---\n\n".join(blocks)
    context = textwrap.shorten(context, width=14000, placeholder="\n…（中略）…")

    system = (
        "あなたは社内資料の要約アシスタントです。"
        "以下の『コンテキスト』内の情報だけを根拠に、日本語で簡潔かつ正確に回答してください。"
        "推測はしないでください。"
        "最後に使用した根拠の番号を角括弧で示してください（例: [1][3]）。"
        "コンテキストに情報が無い場合は、『提供資料からは分かりません』と答えてください。"
    )
    user = (
        f"質問: {query}\n\n"
        "コンテキスト（番号付き）:\n"
        f"{context}\n\n"
        "出力フォーマット:\n"
        "本文（箇条書き歓迎）\n"
        "出典: [番号]"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def answer(query: str, hits: List[Dict[str, Any]]) -> str:
    oa, _ = get_clients()
    msgs = build_messages(query, hits)
    resp = oa.chat.completions.create(model=CHAT_MODEL, messages=msgs, temperature=0.2)
    return resp.choices[0].message.content

# ====== ドキュメント候補（フィルタ用） ======
@st.cache_data
def list_doc_names() -> List[str]:
    names = set()
    if CHUNKS_PATH.exists():
        with CHUNKS_PATH.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    rec = json.loads(line)
                    if dn := rec.get("doc_name"):
                        names.add(dn)
                except Exception:
                    continue
                if i > 100000:  # 念のため上限
                    break
    return sorted(names)

# ====== UI ======
st.set_page_config(page_title="社内PPTX 横断検索AI（RAG）", layout="centered")
st.title("社内PPTX 横断検索AI（RAG）")
st.caption("Step⑤ 仕上げ版：再ランク＋引用つき回答")

with st.sidebar:
    st.subheader("検索設定")
    docs = list_doc_names()
    selected_docs = st.multiselect("対象ドキュメント（任意）", options=docs)
    topk = st.slider("DBからの取得件数 (Top-K)", 5, 20, TOP_K)
    topllm = st.slider("LLMに渡す件数", 3, 10, TOP_K_LLM)
    use_rerank = st.checkbox("Cross-Encoderで再ランクする", value=True)
    st.markdown("---")
    st.caption(f"Embedding: {EMBED_MODEL} / Chat: {CHAT_MODEL}")

query = st.text_input("質問を入力", placeholder="例: ログイン失敗時のエラーメッセージ仕様")
if st.button("検索") and query.strip():
    try:
        oa, coll = get_clients()
        with st.spinner("検索中…"):
            hits = retrieve(oa, coll, query.strip(), k=topk, doc_filter=selected_docs)
            if not hits:
                st.warning("関連チャンクが見つかりませんでした。フィルタやクエリを見直してください。")
            else:
                if use_rerank:
                    hits = rerank(query.strip(), hits, top_n=topllm)
                else:
                    hits = hits[:topllm]

                # GPT回答
                ans = answer(query.strip(), hits)
                # 例: 出典: [1][3] から 1,3 を取る
                cited_ids = sorted({int(m) for m in re.findall(r"\[(\d+)\]", ans) if m.isdigit()})

                # インデックスのズレ対策（1始まり→0始まり）
                used = [hits[i-1] for i in cited_ids if 1 <= i <= len(hits)]
                unused = [h for idx, h in enumerate(hits, start=1) if idx not in cited_ids]
                st.subheader("回答")
                st.write(ans)

                # 根拠一覧
                st.subheader("根拠（引用）")
                if used:
                    st.markdown("**回答で使用された根拠**")
                    for idx in cited_ids:
                        h = hits[idx-1]
                        m = h["meta"]
                        title = f"[{idx}] {m.get('doc_name')} / slide {m.get('slide')} / chunk {m.get('chunk_index')}"
                        with st.expander(title, expanded=True if idx == cited_ids[0] else False):
                            blocks = re.split(r"\n{2,}", h["text"].strip())  # 空行でブロック分け（表/本文の混在に対応）
                            for b in blocks:
                                render_block_with_markdown_and_highlight(b, query.strip())
                else:
                    st.info("回答中に出典番号が見つかりませんでした。プロンプトを見直してください。")

                if unused:
                    with st.expander("その他の候補（参考）", expanded=False):
                        for idx, h in enumerate(hits, start=1):
                            if idx in cited_ids: 
                                continue
                            m = h["meta"]
                            title = f"[{idx}] {m.get('doc_name')} / slide {m.get('slide')} / chunk {m.get('chunk_index')}"
                            with st.expander(title, expanded=False):
                                blocks = re.split(r"\n{2,}", h["text"].strip())  # 空行でブロック分け（表/本文の混在に対応）
                                for b in blocks:
                                    render_block_with_markdown_and_highlight(b, query.strip())

    except Exception as e:
        st.error(str(e))
