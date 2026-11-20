"""Parse a policy document and split it into section-level chunks.

Rules:
- Split on Markdown section headings (`## 4. Title`, `### 4.2 Title`), never by token count.
- One chunk per section/subsection. A heading with no body of its own (only subsections)
  produces no chunk.
- A section longer than `max_words` is split by paragraph (then by sentence if one paragraph
  is still too long). Every piece keeps the section heading via the context header.
- The embedded text is prefixed with a context header:
  "Company: <name> | <doc title> v<version> | §<section_id> <section_title>"
- chunk_id is deterministic: <company_id>_v<version>_s<section_id>_p<part>
"""

import re
from dataclasses import dataclass

TITLE_RE = re.compile(
    r"^#\s+(?P<company>.+?)\s+[—–-]\s+(?P<title>.+?)\s+"
    r"\(Version\s+(?P<version>\d+),\s*effective\s+(?P<date>\d{4}-\d{2}-\d{2})\)\s*$"
)
HEADING_RE = re.compile(r"^(?P<level>#{2,3})\s+(?P<id>\d+(?:\.\d+)*)\.?\s+(?P<title>\S.*?)\s*$")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class ChunkingError(ValueError):
    """The document does not follow the expected policy format."""


@dataclass(frozen=True)
class Section:
    section_id: str
    section_title: str
    body: str


@dataclass(frozen=True)
class ParsedPolicy:
    company_name: str
    doc_title: str
    version: int
    effective_date: str
    sections: list[Section]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    company_id: str
    company_name: str
    version: int
    doc_title: str
    effective_date: str
    section_id: str
    section_title: str
    part: int
    body: str
    text: str  # header + body: what gets embedded and shown to the LLM

    def metadata(self) -> dict[str, str | int]:
        return {
            "chunk_id": self.chunk_id,
            "company_id": self.company_id,
            "company_name": self.company_name,
            "version": self.version,
            "doc_title": self.doc_title,
            "effective_date": self.effective_date,
            "section_id": self.section_id,
            "section_title": self.section_title,
            "part": self.part,
        }


def parse_policy(markdown: str) -> ParsedPolicy:
    lines = markdown.strip().splitlines()
    if not lines:
        raise ChunkingError("Document is empty")

    title = TITLE_RE.match(lines[0].strip())
    if not title:
        raise ChunkingError(
            "First line must be '# <Company> — <Title> (Version N, effective YYYY-MM-DD)'"
        )

    sections: list[Section] = []
    seen: set[str] = set()
    current: tuple[str, str] | None = None
    buf: list[str] = []

    def flush() -> None:
        if current is None:
            if "\n".join(buf).strip():
                raise ChunkingError("Text found before the first section heading")
            return
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(current[0], current[1], body))

    for line in lines[1:]:
        m = HEADING_RE.match(line.strip())
        if m:
            flush()
            sid = m.group("id")
            if sid in seen:
                raise ChunkingError(f"Duplicate section id §{sid}")
            seen.add(sid)
            current = (sid, m.group("title"))
            buf = []
        else:
            buf.append(line)
    flush()

    if not sections:
        raise ChunkingError("No sections found")

    return ParsedPolicy(
        company_name=title.group("company").strip(),
        doc_title=title.group("title").strip(),
        version=int(title.group("version")),
        effective_date=title.group("date"),
        sections=sections,
    )


def _split_words(text: str, max_words: int) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def _pack(units: list[str], max_words: int, sep: str) -> list[str]:
    """Greedily pack units into pieces of at most max_words."""
    pieces: list[str] = []
    cur: list[str] = []
    cur_words = 0
    for unit in units:
        n = len(unit.split())
        if cur and cur_words + n > max_words:
            pieces.append(sep.join(cur))
            cur, cur_words = [], 0
        cur.append(unit)
        cur_words += n
    if cur:
        pieces.append(sep.join(cur))
    return pieces


def split_body(body: str, max_words: int) -> list[str]:
    if len(body.split()) <= max_words:
        return [body]
    units: list[str] = []
    for para in (p.strip() for p in re.split(r"\n\s*\n", body)):
        if not para:
            continue
        if len(para.split()) <= max_words:
            units.append(para)
            continue
        for sentence in SENTENCE_RE.split(para):
            if len(sentence.split()) <= max_words:
                units.append(sentence)
            else:
                units.extend(_split_words(sentence, max_words))
    # Re-pack: paragraphs joined by blank lines; sentence fragments end up joined the same way,
    # which is harmless for retrieval and keeps the logic simple.
    return _pack(units, max_words, "\n\n")


def context_header(company_name: str, doc_title: str, version: int, sid: str, stitle: str) -> str:
    return f"Company: {company_name} | {doc_title} v{version} | §{sid} {stitle}"


def chunk_policy(company_id: str, policy: ParsedPolicy, max_words: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for sec in policy.sections:
        header = context_header(
            policy.company_name, policy.doc_title, policy.version, sec.section_id, sec.section_title
        )
        for part, piece in enumerate(split_body(sec.body, max_words)):
            chunks.append(
                Chunk(
                    chunk_id=f"{company_id}_v{policy.version}_s{sec.section_id}_p{part}",
                    company_id=company_id,
                    company_name=policy.company_name,
                    version=policy.version,
                    doc_title=policy.doc_title,
                    effective_date=policy.effective_date,
                    section_id=sec.section_id,
                    section_title=sec.section_title,
                    part=part,
                    body=piece,
                    text=f"{header}\n\n{piece}",
                )
            )
    return chunks
