"""Engineering-document intake for the SysML agent.

Text extraction (txt/md/pdf/docx), chunking and a keyword retriever. Kept
dependency-light: .docx is read straight out of the zip, and only PDF needs the
optional pypdf package.

Documents live in Postgres keyed by owner, not in a module-level dict: the dict
grew without bound for the life of the process, vanished on restart, and — in a
multi-worker deployment — only existed in whichever worker happened to serve the
upload. Chunks stay as jsonb so one row is one document.
"""

from __future__ import annotations

import io
import re
import uuid
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from psycopg.types.json import Jsonb

from webapp import db

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
TOP_CHUNKS = 4
MAX_CHARS_PER_DOC = 400_000

SUFFIXES = {".txt", ".md", ".markdown", ".sysml", ".kerml", ".pdf", ".docx"}


@dataclass
class Document:
    id: str
    name: str
    chars: int
    chunks: List[str] = field(default_factory=list)


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _docx_lines(node: Any) -> List[str]:
    """Paragraphs as lines, table rows as joined cells."""
    lines: List[str] = []
    for child in node:
        if child.tag == _W + "tbl":
            for row in child.findall(_W + "tr"):
                cells = [
                    " ".join(part.text or "" for part in cell.iter(_W + "t")).strip()
                    for cell in row.findall(_W + "tc")
                ]
                joined = " | ".join(cell for cell in cells if cell)
                if joined:
                    lines.append(joined)
        elif child.tag == _W + "p":
            text = "".join(part.text or "" for part in child.iter(_W + "t")).strip()
            if text:
                lines.append(text)
        else:
            lines.extend(_docx_lines(child))
    return lines


def _docx_text(data: bytes) -> str:
    """Word's own XML: keeps the tables a spec document is made of."""
    from xml.etree import ElementTree

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            xml = archive.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ValueError("not a readable .docx file") from exc
    body = ElementTree.fromstring(xml).find(_W + "body")
    return "\n".join(_docx_lines(body)) if body is not None else ""


def extract_text(name: str, data: bytes) -> str:
    """Plain text from an uploaded engineering document."""
    suffix = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError(
                "PDF reading needs the optional 'pypdf' package, which this "
                "install does not have; paste the text or upload a .docx"
            ) from exc

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".docx":
        return _docx_text(data)

    return data.decode("utf-8", errors="replace")


def chunk_text(text: str) -> List[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def _uuid(value: Any) -> Optional[str]:
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError):
        return None


def _document(row: Dict[str, Any]) -> Document:
    chunks = row.get("chunks")
    return Document(
        id=str(row["id"]),
        name=str(row["name"]),
        chars=int(row.get("chars") or 0),
        chunks=[str(chunk) for chunk in chunks] if isinstance(chunks, list) else [],
    )


def add_document(user_id: Any, name: str, data: bytes) -> Optional[Document]:
    """Extract, chunk and store a document for this owner (None: no owner)."""
    owner = _uuid(user_id)
    if not owner:
        return None
    text = extract_text(name, data)[:MAX_CHARS_PER_DOC]
    chunks = chunk_text(text)
    with db.connection() as conn:
        row = conn.execute(
            "insert into documents (user_id, name, chars, text, chunks) "
            "values (%s, %s, %s, %s, %s) returning id::text as id, name, chars, chunks",
            (owner, name, len(text), text, Jsonb(chunks)),
        ).fetchone()
    return _document(row)


def get_document(user_id: Any, doc_id: Any) -> Optional[Document]:
    owner, wanted = _uuid(user_id), _uuid(doc_id)
    if not owner or not wanted:
        return None
    with db.connection() as conn:
        row = conn.execute(
            "select id::text as id, name, chars, chunks from documents "
            "where id = %s and user_id = %s",
            (wanted, owner),
        ).fetchone()
    return _document(row) if row else None


def list_documents(user_id: Any) -> List[Document]:
    """This owner's documents, oldest first (the order they were attached in)."""
    owner = _uuid(user_id)
    if not owner:
        return []
    with db.connection() as conn:
        rows = conn.execute(
            "select id::text as id, name, chars, chunks from documents "
            "where user_id = %s order by created_at, id",
            (owner,),
        ).fetchall()
    return [_document(row) for row in rows]


def reset_documents(user_id: Any) -> int:
    """Forget this owner's uploads; other tenants keep theirs."""
    owner = _uuid(user_id)
    if not owner:
        return 0
    with db.connection() as conn:
        row = conn.execute(
            "delete from documents where user_id = %s returning 1", (owner,)
        ).fetchone()
    return 1 if row else 0


def _terms(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]{3,}", text.lower())


def search(
    user_id: Any, doc_ids: Sequence[str], query: str, top: int = TOP_CHUNKS
) -> List[Dict[str, str]]:
    """Keyword-scored chunks across the selected documents.

    ponytail: TF overlap, no embeddings - enough to pull the right section of a
    spec, and it keeps uploads instant. Swap in the Chroma store if recall
    matters more than latency.

    Foreign, deleted and malformed ids are simply not in the result set, so a
    client cannot learn about another tenant's upload by naming it.
    """
    owner = _uuid(user_id)
    wanted = [doc for doc in (_uuid(value) for value in doc_ids or []) if doc]
    if not owner or not wanted:
        return []
    with db.connection() as conn:
        rows = conn.execute(
            "select id::text as id, name, chars, chunks from documents "
            "where user_id = %s and id = any(%s::uuid[])",
            (owner, wanted),
        ).fetchall()
    query_terms = set(_terms(query))
    scored = []
    for row in rows:
        document = _document(row)
        for index, chunk in enumerate(document.chunks):
            terms = _terms(chunk)
            if not terms:
                continue
            hits = sum(terms.count(term) for term in query_terms)
            if hits:
                score = hits / (len(terms) ** 0.5)
                scored.append((score, document.name, index, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {"document": name, "chunk": str(index + 1), "text": chunk}
        for _, name, index, chunk in scored[:top]
    ]


def context_block(matches: Iterable[Dict[str, str]]) -> str:
    """Render retrieved chunks as a prompt section."""
    blocks = [
        f"[{match['document']} — chunk {match['chunk']}]\n{match['text']}"
        for match in matches
    ]
    return "\n\n".join(blocks)
