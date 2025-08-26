# qa_cli.py
from __future__ import annotations
from pathlib import Path
import os, textwrap
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from openai import OpenAI

DB_DIR      = Path("vectordb")
COLLECTION  = "pptx_slides"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = os.getenv("CHAT_MODEL", "gpt-4o")  # 必要ならモデルを変更

def get_clients():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY が未設定です（.env または環境変数で設定）")
    oa = OpenAI()
    chroma = chromadb.PersistentClient(path=str(DB_DIR), settings=Settings(allow_reset=False))
    coll = chroma.get_collection(COLLECTION)
    return oa, coll

def embed(oa: OpenAI, text: str) -> List[float]:
    return oa.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding

def retrieve(oa: OpenAI, coll, query: str, k: int = 8, where: Dict[str, Any] | None = None):
    qvec = embed(oa, query)
    res = coll.query(query_embeddings=[qvec], n_results=k, where=where, include=["documents","metadatas","distances"])
    docs   = res["documents"][0]
    metas  = res["metadatas"][0]
    dists  = res.get("distances", [[None]*len(docs)])[0]
    items = []
    for d, m, dist in zip(docs, metas, dists):
        if not d:  # 空テキストはスキップ
            continue
        items.append({"text": d, "meta": m, "score": (1.0 - (dist or 0))})  # ざっくり類似スコア化
    return items

def build_messages(query: str, hits: List[Dict[str, Any]]) -> list:
    """
    コンテキストを番号付きで並べ、回答はその根拠のみで日本語回答させる。
    """
    # コンテキストを安全な長さに整える（長すぎると切る）
    max_chars = 8000
    ctx_blocks = []
    for idx, h in enumerate(hits, start=1):
        meta = h["meta"]
        head = f'[{idx}] {meta.get("doc_name")} / slide {meta.get("slide")} / chunk {meta.get("chunk_index")}'
        body = h["text"].strip()
        if len(body) > max_chars:  # まずは素直に丸める（通常ここまでは行かない）
            body = body[:max_chars]
        ctx_blocks.append(head + "\n" + body)
    context = "\n\n---\n\n".join(ctx_blocks)
    context = textwrap.shorten(context, width=12000, placeholder="\n…（中略）…")  # 念のため

    system = (
        "あなたは社内資料に基づいて回答するアシスタントです。"
        "必ず次の『コンテキスト』内の情報だけを根拠に日本語で簡潔に答えてください。"
        "推測はしないでください。"
        "回答の最後に、使用した根拠の番号を角括弧で並べて示してください（例: [1][3]）。"
        "コンテキストに無い場合は『提供資料からは分かりません』と答えてください。"
    )
    user = (
        f"質問: {query}\n\n"
        "コンテキスト（番号付き）:\n"
        f"{context}\n\n"
        "出力フォーマット:\n"
        "本文（箇条書き歓迎）\n"
        "出典: [番号]を列挙（例: 出典: [2][5])"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def answer(query: str, k: int = 8, where: Dict[str, Any] | None = None):
    oa, coll = get_clients()
    hits = retrieve(oa, coll, query, k=k, where=where)
    if not hits:
        return "関連資料が見つかりませんでした。"

    messages = build_messages(query, hits)
    resp = oa.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2)
    content = resp.choices[0].message.content

    # 出典の対応表を見やすく表示
    legend_lines = []
    for idx, h in enumerate(hits, start=1):
        m = h["meta"]
        legend_lines.append(f"[{idx}] {m.get('doc_name')} / slide {m.get('slide')} / chunk {m.get('chunk_index')}")

    return content + "\n\n" + "— 出典対応表 —\n" + "\n".join(legend_lines)

if __name__ == "__main__":
    import argparse, sys
    from dotenv import load_dotenv
    load_dotenv()  # .env 読み込み（あれば）

    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="*", help="質問文（スペース区切りでOK）")
    p.add_argument("-k", type=int, default=8, help="取得するチャンク数")
    p.add_argument("--doc", action="append", help="doc_name フィルタ（複数指定可）")
    args = p.parse_args()

    q = " ".join(args.query).strip()
    if not q:
        sys.exit("質問文を指定してください。例: python qa_cli.py ログイン 画面 バリデーション 仕様")

    where = {"doc_name": {"$in": args.doc}} if args.doc else None
    print(answer(q, k=args.k, where=where))
