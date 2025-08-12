# embed_chunks.py
from __future__ import annotations
from pathlib import Path
import os, json, hashlib
from typing import List, Dict, Any, Iterable

import chromadb
from chromadb.config import Settings
from openai import OpenAI

# ===== 入出力パス =====
CHUNKS_PATH = Path("chunks.jsonl")          # ステップ②の出力
DB_DIR      = Path("vectordb")              # 永続化ディレクトリ
COLLECTION  = "pptx_slides"                 # コレクション名
BATCH_SIZE  = 64

# ===== OpenAI設定 =====
EMBED_MODEL = "text-embedding-3-small"      # 速くて安い＆十分高精度
# ====================

def load_env():
    # .env があれば読み込む（未インストールでもOKにしたいのでtry）
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def make_id(rec: Dict[str, Any]) -> str:
    """
    （doc名, スライド, チャンクIndex, 先頭50文字）から安定IDを作成
    """
    base = f'{rec.get("doc_name","")}|{rec.get("slide","")}|{rec.get("chunk_index","")}|{rec.get("text","")[:50]}'
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def embed_texts(client: OpenAI, texts: List[str]) -> List[List[float]]:
    """
    OpenAI埋め込み（バッチ）
    """
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def ensure_collection(client: chromadb.Client) -> chromadb.Collection:
    # cosine類似度で作成
    coll = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    return coll

def upsert_chunks():
    load_env()
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"{CHUNKS_PATH} が見つかりません。ステップ②を先に実行してね。")

    # Chroma 初期化（永続モード）
    client = chromadb.PersistentClient(path=str(DB_DIR), settings=Settings(allow_reset=True))
    coll = ensure_collection(client)

    # OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY が未設定です。環境変数または .env に設定してください。")
    oa = OpenAI()

    # 既存IDの重複スキップ用に軽く取得
    existing_ids = set()
    try:
        # 大量だと重いので、ここでは取得しない運用でもOK
        pass
    except Exception:
        pass

    batch_docs: List[str] = []
    batch_metas: List[Dict[str, Any]] = []
    batch_ids:   List[str] = []

    count = 0
    for rec in read_jsonl(CHUNKS_PATH):
        doc = rec["text"]
        _id = make_id(rec)
        if _id in existing_ids:
            continue

        meta = {
            "doc_name": rec.get("doc_name"),
            "path": rec.get("path"),
            "slide": rec.get("slide"),
            "chunk_index": rec.get("chunk_index"),
            "char_start": rec.get("char_start"),
            "char_end": rec.get("char_end"),
        }

        batch_docs.append(doc)
        batch_metas.append(meta)
        batch_ids.append(_id)

        if len(batch_docs) >= BATCH_SIZE:
            embeddings = embed_texts(oa, batch_docs)
            coll.upsert(documents=batch_docs, metadatas=batch_metas, ids=batch_ids, embeddings=embeddings)
            count += len(batch_docs)
            batch_docs, batch_metas, batch_ids = [], [], []
            print(f"…indexed {count}")

    # 端数
    if batch_docs:
        embeddings = embed_texts(oa, batch_docs)
        coll.upsert(documents=batch_docs, metadatas=batch_metas, ids=batch_ids, embeddings=embeddings)
        count += len(batch_docs)
        print(f"…indexed {count}")

    print(f"✅ 完了。合計 {count} チャンクを {DB_DIR}/ に保存しました。")

def quick_query(q: str, k: int = 5, where: Dict[str, Any] | None = None):
    """
    ベクトルDBのクイック動作確認。
    """
    load_env()
    client = chromadb.PersistentClient(path=str(DB_DIR))
    coll = client.get_collection(COLLECTION)

    oa = OpenAI()
    q_emb = embed_texts(oa, [q])[0]

    res = coll.query(query_embeddings=[q_emb], n_results=k, where=where)
    # 表示
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0] or res.get("embeddings", [[]])[0]
    print("---- Query Result ----")
    for i, (text, meta) in enumerate(zip(docs, metas), start=1):
        head = text[:120].replace("\n", " ")
        print(f"[{i}] {meta.get('doc_name')}  slide:{meta.get('slide')}  chunk:{meta.get('chunk_index')}")
        print(f"    {head}…")
    print("----------------------")

if __name__ == "__main__":
    # 1) インデックス作成
    upsert_chunks()

    # 2) 動作確認（任意のクエリを入れてテスト）
    #    例: quick_query("ログイン画面のバリデーション仕様")
    # quick_query("ここに質問を書く")
