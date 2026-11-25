"""Document intake: .docx must be readable without python-docx installed.

Extraction is pure and stays that way; storage is now per owner in Postgres, so
the same tests cover both halves of the module.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from webapp import documents

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
GEAR = "Regalbedienger\u00e4t"
LIFT = "Hubh\u00f6he"
PARTS = "Teile & St\u00fccklisten"


def para(text: str) -> str:
    return f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"


def cell(text: str) -> str:
    return f"<w:tc>{para(text)}</w:tc>"


def make_docx(paragraphs, table=None) -> bytes:
    body = "".join(para(text) for text in paragraphs)
    if table:
        rows = "".join(
            "<w:tr>" + "".join(cell(value) for value in row) + "</w:tr>" for row in table
        )
        body += f"<w:tbl>{rows}</w:tbl>"
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<w:document xmlns:w="{W}"><w:body>{body}</w:body></w:document>'
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", xml)
    return buffer.getvalue()


# --- extraction ------------------------------------------------------------


def test_docx_paragraphs_are_extracted_without_python_docx():
    data = make_docx(["Beschreibung Automatisches Kleinteilelager", f"{GEAR} RBG-1"])

    text = documents.extract_text("Beschreibung.docx", data)

    assert text.splitlines() == ["Beschreibung Automatisches Kleinteilelager", f"{GEAR} RBG-1"]


def test_docx_table_rows_survive_as_one_line_per_row():
    data = make_docx(
        ["Anforderungen"],
        table=[["Merkmal", "Wert"], ["Traglast", "50 kg"], [LIFT, "12 m"]],
    )

    lines = documents.extract_text("spec.docx", data).splitlines()

    assert lines == ["Anforderungen", "Merkmal | Wert", "Traglast | 50 kg", f"{LIFT} | 12 m"]


def test_docx_xml_entities_and_empty_paragraphs_are_handled():
    data = make_docx(["Last &amp; hub &lt;12m&gt;", ""])

    assert documents.extract_text("entities.docx", data).splitlines() == ["Last & hub <12m>"]


def test_a_zip_without_word_xml_is_reported_as_unreadable():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("hello.txt", "not a document")

    with pytest.raises(ValueError, match="not a readable .docx"):
        documents.extract_text("broken.docx", buffer.getvalue())


def test_pdf_without_pypdf_says_what_is_missing():
    try:
        import pypdf  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("pypdf is installed here")

    with pytest.raises(ValueError, match="pypdf"):
        documents.extract_text("plan.pdf", b"%PDF-1.4")


def test_plain_text_upload_still_decodes():
    assert documents.extract_text("notes.md", PARTS.encode()) == PARTS


def test_chunking_overlaps_so_a_cut_sentence_is_not_lost():
    text = "a" * (documents.CHUNK_CHARS + 50)

    chunks = documents.chunk_text(text)

    assert len(chunks) == 2
    assert len(chunks[0]) == documents.CHUNK_CHARS
    assert chunks[1].startswith(chunks[0][-documents.CHUNK_OVERLAP:])


# --- storage: one owner's documents are invisible to another ----------------


async def test_add_document_chunks_the_extracted_text(user_a):
    document = documents.add_document(
        user_a.id, "angebot.docx", make_docx([GEAR] * 3)
    )

    assert document.chars > 0
    assert document.chunks
    assert GEAR in document.chunks[0]


async def test_documents_are_listed_per_owner(user_a, user_b):
    mine = documents.add_document(user_a.id, "a.txt", b"Traglast 50 kg")

    assert [d.id for d in documents.list_documents(user_a.id)] == [mine.id]
    assert documents.list_documents(user_b.id) == []
    assert documents.get_document(user_b.id, mine.id) is None
    assert documents.get_document(user_a.id, mine.id).name == "a.txt"


async def test_search_ignores_a_foreign_document_id(user_a, user_b):
    mine = documents.add_document(user_a.id, "spec.txt", b"Traglast 50 kg hub 12 m")

    assert documents.search(user_a.id, [mine.id], "Traglast")
    assert documents.search(user_b.id, [mine.id], "Traglast") == []
    # A malformed id is not an error either.
    assert documents.search(user_a.id, ["not-a-uuid"], "Traglast") == []


async def test_reset_only_touches_the_caller(user_a, user_b):
    mine = documents.add_document(user_a.id, "a.txt", b"one")
    theirs = documents.add_document(user_b.id, "b.txt", b"two")

    documents.reset_documents(user_a.id)

    assert documents.list_documents(user_a.id) == []
    assert [d.id for d in documents.list_documents(user_b.id)] == [theirs.id]
    assert documents.get_document(user_a.id, mine.id) is None


async def test_a_foreign_document_is_not_retrievable_over_http(user_a, user_b):
    uploaded = await user_a.client.post(
        "/api/documents", files=[("files", ("plan.txt", b"Hub 12 m", "text/plain"))]
    )
    assert uploaded.status_code == 200
    document_id = uploaded.json()["documents"][0]["id"]

    assert (await user_b.client.get("/api/documents")).json()["documents"] == []
    assert documents.get_document(user_b.id, document_id) is None


async def test_an_unsupported_type_is_refused(user_a):
    response = await user_a.client.post(
        "/api/documents", files=[("files", ("run.exe", b"MZ", "application/octet-stream"))]
    )

    assert response.status_code == 400
    assert "unsupported file type" in response.json()["detail"]
