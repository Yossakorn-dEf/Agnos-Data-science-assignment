
import os, re, pandas as pd, faiss, numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

def build_index_from_html(raw_dir, index_dir, model_name="BAAI/bge-small-en-v1.5"):
    os.makedirs(index_dir, exist_ok=True)

    def read_meta_block(html_text):
        m = re.search(r"<!--\s*META:(.*?)-->", html_text, flags=re.S)
        meta = {}
        if m:
            for line in m.group(1).splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
        return meta

    def parse_thread_html(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            html = f.read()
        meta = read_meta_block(html)
        soup = BeautifulSoup(html, "html.parser")
        posts = soup.select("article, div.post, div.message")
        rows = []
        for p in posts:
            text = p.get_text(" ", strip=True)
            if not text or len(text) < 20: continue
            author = p.select_one(".username")
            date   = p.select_one("time")
            rows.append({
                "file": os.path.basename(filepath),
                "title": meta.get("title", ""),
                "url": meta.get("source_url", ""),
                "scraped_at": meta.get("scraped_at", ""),
                "author": author.get_text(strip=True) if author else None,
                "date": date.get_text(strip=True) if date else None,
                "content": text
            })
        return rows

    # รวมข้อมูลจากไฟล์ .html
    all_rows = []
    for fn in sorted(os.listdir(raw_dir)):
        if not fn.endswith(".html"): continue
        if fn.startswith(("000_home", "home_")): continue
        all_rows.extend(parse_thread_html(os.path.join(raw_dir, fn)))

    if not all_rows:
        print("⚠️ ไม่พบโพสต์ในไฟล์ HTML")
        return

    df = pd.DataFrame(all_rows)
    df["content_clean"] = df["content"].apply(lambda x: " ".join(x.split()))
    df = df[df["content_clean"].str.len() > 0].reset_index(drop=True)

    # Chunk
    def chunk_text(text, chunk_size=600, overlap=100):
        words, out, i = text.split(), [], 0
        while i < len(words):
            out.append(" ".join(words[i:i+chunk_size]))
            i += max(1, chunk_size - overlap)
        return out

    chunks = []
    for i, r in df.iterrows():
        for j, ch in enumerate(chunk_text(r["content_clean"])):
            chunks.append({
                "doc_id": i,
                "chunk_id": j,
                "title": r["title"],
                "url": r["url"],
                "author": r["author"],
                "date": r["date"],
                "content_chunk": ch
            })

    df_chunks = pd.DataFrame(chunks)

    # Embedding
    embedder = SentenceTransformer(model_name)
    embs = embedder.encode(df_chunks["content_chunk"].tolist(),
                           batch_size=32, show_progress_bar=True)

    # FAISS index
    d = embs.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embs, dtype="float32"))

    # Save
    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    df_chunks.to_parquet(os.path.join(index_dir, "meta.parquet"), index=False)

    print(f" Build index เสร็จ | posts={len(df)} | chunks={len(df_chunks)}")
    print(f"   -> {os.path.join(index_dir, 'faiss.index')}")
    print(f"   -> {os.path.join(index_dir, 'meta.parquet')}")
