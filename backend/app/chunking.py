"""
Document loading + chunking strategy.

Design notes (the kind of thing an interviewer will ask about):
- We chunk by *tokens*, not characters, using tiktoken's cl100k_base encoding
  as a stable proxy tokenizer. This keeps chunk sizes consistent for the
  embedding model's context window regardless of how verbose the prose is.
- We use a *recursive* splitter: try to break on paragraph boundaries first,
  then sentences, then words, only falling back to a hard token cut if a
  single unit is still too large (e.g. a giant table dumped as one "paragraph").
  This keeps chunks semantically coherent instead of slicing mid-sentence.
- We keep a token overlap between consecutive chunks so that facts sitting
  near a chunk boundary aren't stranded without surrounding context.
- Every chunk carries its source document, page (when known), and position,
  because citations are only as good as the metadata attached to them.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tiktoken
from pypdf import PdfReader
from docx import Document as DocxDocument

from .models import Chunk

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


@dataclass
class RawPage:
    """A page (or the whole doc, for formats without pages) of raw text."""

    text: str
    page: Optional[int]


# --------------------------------------------------------------------------
# Loaders: turn a file on disk into a list of (text, page) units
# --------------------------------------------------------------------------

def load_pdf(path: Path) -> list[RawPage]:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(RawPage(text=text, page=i + 1))
    return pages


def load_docx(path: Path) -> list[RawPage]:
    doc = DocxDocument(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return [RawPage(text="\n\n".join(paragraphs), page=None)]


def load_text(path: Path) -> list[RawPage]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [RawPage(text=text, page=None)]


LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_text,
    ".md": load_text,
}


def load_document(path: Path) -> list[RawPage]:
    ext = path.suffix.lower()
    if ext not in LOADERS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {sorted(LOADERS.keys())}"
        )
    return LOADERS[ext](path)


# --------------------------------------------------------------------------
# Recursive, token-aware splitter
# --------------------------------------------------------------------------

_SEPARATORS = ["\n\n", "\n", r"(?<=[.!?]) ", " "]  # paragraph -> line -> sentence -> word


def _split_on(text: str, separator: str) -> list[str]:
    if separator == " ":
        return text.split(" ")
    parts = re.split(separator, text)
    return [p for p in parts if p != ""]


def _recursive_split(text: str, max_tokens: int, seps: list[str]) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text]
    if not seps:
        # Hard fallback: cut by raw token slices.
        tokens = _ENCODING.encode(text)
        return [
            _ENCODING.decode(tokens[i : i + max_tokens])
            for i in range(0, len(tokens), max_tokens)
        ]

    sep, rest_seps = seps[0], seps[1:]
    pieces = _split_on(text, sep)
    joiner = "" if sep == " " else (sep if sep in ("\n\n", "\n") else " ")

    chunks: list[str] = []
    buffer = ""
    for piece in pieces:
        candidate = (buffer + joiner + piece) if buffer else piece
        if count_tokens(candidate) <= max_tokens:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer)
            if count_tokens(piece) > max_tokens:
                chunks.extend(_recursive_split(piece, max_tokens, rest_seps))
                buffer = ""
            else:
                buffer = piece
    if buffer:
        chunks.append(buffer)
    return chunks


def _add_overlap(pieces: list[str], overlap_tokens: int) -> list[str]:
    if overlap_tokens <= 0 or len(pieces) <= 1:
        return pieces
    overlapped = [pieces[0]]
    for prev, curr in zip(pieces, pieces[1:]):
        prev_tokens = _ENCODING.encode(prev)
        tail = _ENCODING.decode(prev_tokens[-overlap_tokens:]) if prev_tokens else ""
        overlapped.append((tail + " " + curr).strip() if tail else curr)
    return overlapped


def chunk_pages(
    pages: list[RawPage],
    doc_id: str,
    source: str,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    for raw_page in pages:
        pieces = _recursive_split(raw_page.text, chunk_size_tokens, _SEPARATORS)
        pieces = [p.strip() for p in pieces if p.strip()]
        pieces = _add_overlap(pieces, chunk_overlap_tokens)
        for piece in pieces:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_{uuid.uuid4().hex[:8]}",
                    doc_id=doc_id,
                    source=source,
                    chunk_index=idx,
                    page=raw_page.page,
                    text=piece,
                )
            )
            idx += 1
    return chunks


def load_and_chunk_document(
    path: Path,
    doc_id: str,
    source: str,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> list[Chunk]:
    pages = load_document(path)
    if not pages:
        raise ValueError(f"No extractable text found in '{source}'.")
    return chunk_pages(pages, doc_id, source, chunk_size_tokens, chunk_overlap_tokens)
