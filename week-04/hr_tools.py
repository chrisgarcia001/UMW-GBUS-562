"""The logic behind hr_mcp_server.py: HR document search and leave balance lookup.

Search follows week-03/ask.py: one chunk per document section, each starting
"<document title> > <section heading>", stored in a Chroma database using Chroma's
built-in embedding model and cosine similarity.

Nothing here may print to stdout. The MCP server's stdio transport uses stdout for its
protocol messages, so all logging goes to stderr.
"""

import csv
import hashlib
import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "hartwell-hr"
DOCS_DIR = DATA_DIR / "hr-docs"
EMPLOYEES_CSV = DATA_DIR / "employees.csv"
DB_DIR = HERE / "chroma_db"

COLLECTION = "hartwell_hr_docs"
TOP_K = 4

log = logging.getLogger("hr_tools")


class EmployeeRecordError(Exception):
    """No single, usable employee record could be found. The message is shown to the model."""


# ---------------------------------------------------------------------------
# Leave balances
# ---------------------------------------------------------------------------


def leave_balance(email):
    """Return the leave balance for an exact work email match (ignoring case)."""
    wanted = email.strip().lower()
    do_not_state = "Do not state or estimate a leave balance for this person."
    try:
        with EMPLOYEES_CSV.open(encoding="utf-8-sig", newline="") as f:
            matches = [
                row for row in csv.DictReader(f)
                if wanted and (row.get("email") or "").strip().lower() == wanted
            ]
        if not matches:
            raise EmployeeRecordError(f"No employee record matches '{email}'. {do_not_state}")
        if len(matches) > 1:
            raise EmployeeRecordError(f"More than one employee record matches '{email}'. {do_not_state}")
        row = matches[0]
        return {
            "name": row["name"],
            "vacation_days_left": int(row["vacation_days_left"]),
            "sick_days_left": int(row["sick_days_left"]),
            "personal_days_left": int(row["personal_days_left"]),
        }
    except (OSError, KeyError, TypeError, ValueError) as e:
        log.error("Could not read a balance from %s: %r", EMPLOYEES_CSV.name, e)
        raise EmployeeRecordError(f"The employee record for '{email}' could not be read. {do_not_state}") from e


# ---------------------------------------------------------------------------
# Reading documents into chunks
# ---------------------------------------------------------------------------


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def read_html(path):
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")

    title = clean(soup.title.get_text()).split(" | ")[0] if soup.title else ""
    h1 = soup.find("h1")
    if not title and h1:
        title = clean(h1.get_text())
    title = title or path.stem

    meta = soup.find(class_="meta")
    version_note = clean(meta.get_text()) if meta else ""
    if meta:
        meta.decompose()

    # The footer only says the company is fictional, so it is not indexed as policy text.
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    root = soup.find("main") or soup.body or soup

    sections = []
    h2 = None
    for el in root.find_all(["h1", "h2", "h3", "p", "ul", "ol", "table"]):
        # Paragraphs inside lists and tables are handled with their list or table.
        if el.find_parent(["ul", "ol", "table"]) or el.name == "h1":
            continue
        if el.name == "h2":
            h2 = clean(el.get_text())
            sections.append((h2, []))
            continue
        if el.name == "h3":
            h3 = clean(el.get_text())
            sections.append((f"{h2} > {h3}" if h2 else h3, []))
            continue

        if el.name in ("ul", "ol"):
            text = "\n".join("- " + clean(li.get_text()) for li in el.find_all("li", recursive=False))
        elif el.name == "table":
            text = "\n".join(
                " | ".join(clean(c.get_text()) for c in tr.find_all(["th", "td"]))
                for tr in el.find_all("tr")
            )
        else:
            text = clean(el.get_text())
        if not text:
            continue
        if not sections:
            sections.append(("Overview", []))
        sections[-1][1].append(text)

    return title, version_note, sections


def build_chunks(files):
    chunks = []
    for path in files:
        title, version_note, sections = read_html(path)
        doc_chunks = [
            {
                "text": f"{title} > {heading}\n\n" + "\n".join(paragraphs),
                "source": path.name,
                "title": title,
                "section": heading,
                "version_note": version_note,
            }
            for heading, paragraphs in sections
            if paragraphs
        ]
        if not doc_chunks:
            log.warning("No text found in %s.", path.name)
        log.info("  %s: %d chunks", path.name, len(doc_chunks))
        chunks.extend(doc_chunks)
    return chunks


def doc_files():
    return sorted(DOCS_DIR.glob("*.html"))


def fingerprint(files):
    """Changes whenever a document is added, removed, or edited."""
    h = hashlib.sha256()
    for p in files:
        st = p.stat()
        h.update(f"{p.name}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Chroma database
# ---------------------------------------------------------------------------


def open_db():
    # Imported here because chromadb is slow to load. The server can then start quickly,
    # and a session that only looks up balances never pays for it.
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=str(DB_DIR), settings=Settings(anonymized_telemetry=False))
    return client, embedding_functions.DefaultEmbeddingFunction()


def rebuild_index(client=None, embed=None):
    """Replace the database with a fresh index of the documents. Returns the collection."""
    if client is None:
        client, embed = open_db()
    files = doc_files()
    log.info("Indexing %s/ ...", DOCS_DIR.name)
    chunks = build_chunks(files)
    if not chunks:
        raise RuntimeError(f"No document text found in {DOCS_DIR}.")
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    collection = client.create_collection(
        COLLECTION,
        embedding_function=embed,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"fingerprint": fingerprint(files)},
    )
    collection.add(
        ids=[f"{c['source']}::{i}" for i, c in enumerate(chunks)],
        documents=[c["text"] for c in chunks],
        metadatas=[{k: c[k] for k in ("source", "title", "section", "version_note")} for c in chunks],
    )
    log.info("Stored %d chunks in %s/.", len(chunks), DB_DIR.name)
    return collection


def current_collection():
    """Open the index, building it if it's missing or the documents have changed."""
    client, embed = open_db()
    if COLLECTION in [c.name for c in client.list_collections()]:
        collection = client.get_collection(COLLECTION, embedding_function=embed)
        if (collection.metadata or {}).get("fingerprint") == fingerprint(doc_files()):
            return collection
        log.info("The documents have changed; rebuilding the index.")
    return rebuild_index(client, embed)


def search_docs(query):
    """Return the TOP_K chunks closest to the query, each with its source file."""
    res = current_collection().query(query_texts=[query], n_results=TOP_K)
    return [
        {
            "source": meta["source"],
            "title": meta["title"],
            "section": meta["section"],
            "version_note": meta["version_note"],
            "similarity": round(1 - dist, 3),
            "text": doc,
        }
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
