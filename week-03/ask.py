"""Ask questions about the documents in copperline-docs/ using retrieval-augmented generation.

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-03\\ask.py "How much can I spend on dinner in Boston?"
    .venv\\Scripts\\python.exe week-03\\ask.py              (interactive: ask several questions)
    .venv\\Scripts\\python.exe week-03\\ask.py --rebuild    (re-index after adding or removing documents)

Each document is split into one chunk per section. Every chunk starts with
"<document title> > <section heading>" and records the file it came from. Chunks are
stored in a Chroma database (week-03/chroma_db/) using Chroma's built-in embedding model.
For each question, the 4 most similar chunks are sent to Claude, which answers only
from them and cites the source file for each fact.
"""

import argparse
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

import anthropic
import chromadb
import pymupdf
from bs4 import BeautifulSoup
from chromadb.config import Settings
from chromadb.utils import embedding_functions

HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE / "copperline-docs"
DB_DIR = HERE / "chroma_db"
# API-Key.txt lives in the folder that contains the repository (never inside it).
API_KEY_FILE = HERE.parent.parent / "API-Key.txt"

COLLECTION = "copperline_docs"
MODEL = "claude-sonnet-5"
TOP_K = 4
SNIPPET_CHARS = 150
MAX_CHUNK_CHARS = 2000  # longer sections are split at paragraph breaks

SUPPORTED = {".html", ".htm", ".pdf", ".md", ".txt"}

# Facts a section depends on but that are only stated elsewhere in the same document.
# Keyed by file name, then section heading ("*" = every section except the one that
# defines the fact). Add entries here when a new document has the same problem.
CONTEXT_NOTES = {
    "month-end-close-calendar.html": {
        "*": (
            "How the business days in this calendar are counted: business day 1 is the "
            "first business day of the new month, and business day minus 1 is the last "
            "business day of the month being closed."
        ),
    },
    "travel-expense-policy.html": {
        "Hotels": (
            "City tiers in this policy: Tier 1 cities are New York, San Francisco, Boston, "
            "Chicago, Washington DC, Los Angeles, and Seattle. Tier 2 cities are every "
            "other city in the United States."
        ),
        "Alcohol": "Client entertainment rules are in the Client entertainment section of this policy.",
    },
}
# The section that states each "*" note itself, so the note isn't repeated there.
NOTE_SOURCE_SECTIONS = {"month-end-close-calendar.html": "How the close calendar works"}

SYSTEM_PROMPT = """You answer questions about Copperline Manufacturing's finance documents.

Rules:
- Answer only from the sources in the user's message. Do not use outside knowledge or make assumptions beyond what the sources say.
- Cite the source file in square brackets after each fact, for example [travel-expense-policy.html].
- If the sources do not contain the answer, say so plainly instead of guessing. If they answer only part of the question, answer that part and say what the sources don't cover.
- Keep the answer short and direct."""


# ---------------------------------------------------------------------------
# Reading documents into sections
# ---------------------------------------------------------------------------
# Every reader returns a dict:
#   {"title": str, "version_note": str, "contact": str,
#    "sections": [(heading, [paragraph, ...]), ...]}


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def read_html(path):
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")

    title = clean(soup.title.get_text()) if soup.title else ""
    title = title.split(" | ")[0]  # drop the " | Copperline Manufacturing" site suffix
    h1 = soup.find("h1")
    if not title and h1:
        title = clean(h1.get_text())

    footer = soup.find("footer")
    contact = clean(footer.get_text()) if footer else ""

    meta = soup.find(class_="meta")
    version_note = clean(meta.get_text()) if meta else ""

    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    if meta:
        meta.decompose()
    root = soup.find("main") or soup.body or soup

    sections = []
    h2 = h3 = None
    for el in root.find_all(["h1", "h2", "h3", "p", "ul", "ol", "table"]):
        # Paragraphs inside lists and tables are handled with their list or table.
        if el.find_parent(["ul", "ol", "table"]):
            continue
        if el.name == "h1":
            continue
        if el.name == "h2":
            h2, h3 = clean(el.get_text()), None
            sections.append((h2, []))
            continue
        if el.name == "h3":
            h3 = clean(el.get_text())
            sections.append((f"{h2} > {h3}" if h2 else h3, []))
            continue

        if el.name in ("ul", "ol"):
            text = "\n".join("- " + clean(li.get_text()) for li in el.find_all("li", recursive=False))
        elif el.name == "table":
            rows = [
                " | ".join(clean(c.get_text()) for c in tr.find_all(["th", "td"]))
                for tr in el.find_all("tr")
            ]
            text = "\n".join(rows)
        else:
            text = clean(el.get_text())
        if not text:
            continue
        if not sections:
            sections.append(("Overview", []))
        sections[-1][1].append(text)

    return {"title": title, "version_note": version_note, "contact": contact, "sections": sections}


def read_pdf(path):
    """Headings are found by font: bold lines larger than the body text."""
    doc = pymupdf.open(path)
    title = clean((doc.metadata or {}).get("title", "")).split(" | ")[0]

    # Collect each text block (a paragraph) with the size and boldness of its first span.
    blocks = []
    for page in doc:
        height = page.rect.height
        for b in page.get_text("dict")["blocks"]:
            lines = b.get("lines", [])
            if not lines:
                continue
            first = lines[0]["spans"][0]
            text = clean(" ".join("".join(s["text"] for s in l["spans"]) for l in lines))
            if text:
                blocks.append({
                    "text": text,
                    "size": round(first["size"], 1),
                    "bold": bool(first["flags"] & 16) or "bold" in first["font"].lower(),
                    "top": b["bbox"][1] / height,
                    "bottom": b["bbox"][3] / height,
                })
    doc.close()
    if not blocks:
        return {"title": title or path.stem, "version_note": "", "contact": "", "sections": []}

    # Body size = the size holding the most text. Smaller text, or text in the top or
    # bottom 5% of the page, is page furniture (running header, footer, page numbers).
    sizes = Counter()
    for b in blocks:
        sizes[b["size"]] += len(b["text"])
    body = sizes.most_common(1)[0][0]
    blocks = [
        b for b in blocks
        if b["size"] >= body and b["top"] > 0.05 and b["bottom"] < 0.95
        and not re.fullmatch(r"(Page )?\d+( of \d+)?", b["text"])
    ]

    # Section headings = the most common size among bold text larger than the body.
    # Anything larger still is the document title or a banner.
    big = Counter(b["size"] for b in blocks if b["bold"] and b["size"] > body)
    heading = min(big.most_common(), key=lambda kv: (-kv[1], kv[0]))[0] if big else None

    sections, front = [], []
    for b in blocks:
        if heading and b["bold"] and b["size"] == heading:
            sections.append((b["text"], []))
        elif heading and b["size"] > heading:
            if not title:
                title = b["text"]  # first title-sized line, if the PDF has no title metadata
        elif sections:
            sections[-1][1].append(b["text"])
        else:
            front.append(b["text"])  # text before the first heading: department, version

    if not sections and front:
        sections = [("Overview", front)]
        front = []
    return {"title": title or path.stem, "version_note": " ".join(front), "contact": "", "sections": sections}


def read_markdown(path):
    title, sections = "", []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m and len(m.group(1)) == 1 and not title:
            title = clean(m.group(2))
        elif m:
            sections.append((clean(m.group(2)), [""]))
        elif line.strip():
            if not sections:
                sections.append(("Overview", [""]))
            sections[-1][1][-1] += " " + line.strip()
        elif sections and sections[-1][1][-1]:
            sections[-1][1].append("")  # blank line = new paragraph
    sections = [(h, [clean(p) for p in ps if p.strip()]) for h, ps in sections]
    return {"title": title or path.stem, "version_note": "", "contact": "", "sections": sections}


def read_text(path):
    paragraphs = [clean(p) for p in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")) if p.strip()]
    return {"title": path.stem.replace("-", " ").title(), "version_note": "", "contact": "",
            "sections": [("Full text", paragraphs)]}


READERS = {".html": read_html, ".htm": read_html, ".pdf": read_pdf, ".md": read_markdown, ".txt": read_text}


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def split_long(paragraphs):
    """Group paragraphs into parts of at most MAX_CHUNK_CHARS (a paragraph is never cut)."""
    parts, current = [], []
    for p in paragraphs:
        if current and len("\n".join(current + [p])) > MAX_CHUNK_CHARS:
            parts.append(current)
            current = []
        current.append(p)
    if current:
        parts.append(current)
    return parts


def chunk_document(path, doc):
    notes = CONTEXT_NOTES.get(path.name, {})
    chunks = []
    for heading, paragraphs in doc["sections"]:
        if not paragraphs:
            continue
        note = notes.get(heading)
        if note is None and heading != NOTE_SOURCE_SECTIONS.get(path.name):
            note = notes.get("*")
        parts = split_long(paragraphs)
        for n, part in enumerate(parts, start=1):
            label = heading if len(parts) == 1 else f"{heading} (part {n} of {len(parts)})"
            body = "\n".join(([f"({note})"] if note else []) + part)
            chunks.append({
                "text": f"{doc['title']} > {label}\n\n{body}",
                "source": path.name,
                "title": doc["title"],
                "section": label,
                "version_note": doc["version_note"],
            })
    return chunks


def doc_files():
    return sorted(p for p in DOCS_DIR.iterdir() if p.is_file())


def fingerprint(files):
    """Changes whenever a document is added, removed, or edited."""
    h = hashlib.sha256()
    for p in files:
        st = p.stat()
        h.update(f"{p.name}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def build_chunks(files):
    chunks, seen_contacts = [], set()
    for path in files:
        ext = path.suffix.lower()
        if ext not in SUPPORTED:
            print(f"  Skipping {path.name}: {ext or 'no extension'} files are not supported "
                  f"(supported: {', '.join(sorted(SUPPORTED))}).")
            continue
        doc = READERS[ext](path)
        doc_chunks = chunk_document(path, doc)
        # The same contact footer appears on every page; store each distinct one once.
        if doc["contact"] and doc["contact"] not in seen_contacts:
            seen_contacts.add(doc["contact"])
            doc_chunks.append({
                "text": f"{doc['title']} > Contact\n\n{doc['contact']}",
                "source": path.name, "title": doc["title"], "section": "Contact",
                "version_note": doc["version_note"],
            })
        if not doc_chunks:
            print(f"  Warning: no text found in {path.name}.")
        print(f"  {path.name}: {len(doc_chunks)} chunks")
        chunks.extend(doc_chunks)
    return chunks


# ---------------------------------------------------------------------------
# Chroma database
# ---------------------------------------------------------------------------


def open_db():
    client = chromadb.PersistentClient(path=str(DB_DIR), settings=Settings(anonymized_telemetry=False))
    return client, embedding_functions.DefaultEmbeddingFunction()


def rebuild(client, embed):
    files = doc_files()
    print(f"Indexing {DOCS_DIR.name}/ ...")
    chunks = build_chunks(files)
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    collection = client.create_collection(
        COLLECTION,
        embedding_function=embed,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"fingerprint": fingerprint(files)},
    )
    if chunks:
        collection.add(
            ids=[f"{c['source']}::{i}" for i, c in enumerate(chunks)],
            documents=[c["text"] for c in chunks],
            metadatas=[{k: c[k] for k in ("source", "title", "section", "version_note")} for c in chunks],
        )
    print(f"Stored {len(chunks)} chunks in {DB_DIR.name}/.\n")
    return collection


def get_collection(client, embed):
    if COLLECTION not in [c.name for c in client.list_collections()]:
        return rebuild(client, embed)
    collection = client.get_collection(COLLECTION, embedding_function=embed)
    if (collection.metadata or {}).get("fingerprint") != fingerprint(doc_files()):
        print("Note: the documents have changed since the database was built. "
              "Run with --rebuild to update it.\n")
    return collection


# ---------------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------------


def retrieve(collection, question):
    res = collection.query(query_texts=[question], n_results=TOP_K)
    return [
        {"text": doc, "source": meta["source"], "similarity": 1 - dist}
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def print_hits(hits):
    print(f"Top {len(hits)} chunks:")
    for rank, h in enumerate(hits, start=1):
        snippet = clean(h["text"])[:SNIPPET_CHARS]
        print(f"  {rank}. similarity {h['similarity']:.3f}  {h['source']}")
        print(f"     {snippet}...")
    print()


def ask_claude(client, question, hits):
    sources = "\n\n".join(
        f'<source file="{h["source"]}">\n{h["text"]}\n</source>' for h in hits
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"<sources>\n{sources}\n</sources>\n\nQuestion: {question}",
        }],
    )
    if response.stop_reason == "refusal":
        return "(Claude declined to answer this question.)"
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if response.stop_reason == "max_tokens":
        text += "\n(Answer cut off at the length limit.)"
    return text


def load_api_key():
    try:
        key = API_KEY_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        sys.exit(f"API key file not found: {API_KEY_FILE}")
    if not key:
        sys.exit(f"API key file is empty: {API_KEY_FILE}")
    return key


def answer(collection, claude, question):
    hits = retrieve(collection, question)
    print()
    print_hits(hits)
    try:
        print("Answer:\n" + ask_claude(claude, question, hits) + "\n")
    except anthropic.AuthenticationError:
        print(f"Claude rejected the API key in {API_KEY_FILE.name}.\n")
    except anthropic.NotFoundError:
        print(f"Model {MODEL} was not found.\n")
    except anthropic.RateLimitError:
        print("Rate limited by the API. Wait a moment and try again.\n")
    except anthropic.APIStatusError as e:
        print(f"API error {e.status_code}: {e.message}\n")
    except anthropic.APIConnectionError:
        print("Could not reach the Claude API. Check your internet connection.\n")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Answer questions from the documents in copperline-docs/.")
    parser.add_argument("question", nargs="*", help="the question (leave out to ask several interactively)")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild the database after adding or removing documents")
    args = parser.parse_args()

    db, embed = open_db()
    collection = rebuild(db, embed) if args.rebuild else get_collection(db, embed)
    if args.rebuild and not args.question:
        return

    claude = anthropic.Anthropic(api_key=load_api_key())

    if args.question:
        answer(collection, claude, " ".join(args.question))
        return

    print("Ask a question about the documents (press Enter on an empty line to quit).")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break
        answer(collection, claude, question)


if __name__ == "__main__":
    main()
