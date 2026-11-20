import pytest

from app.ingest.chunker import ChunkingError, chunk_policy, parse_policy
from tests.conftest import read_policy

EXPECTED = {
    ("nimbus", "seed/nimbus/v1.md"): (1, 21),
    ("kestrel", "seed/kestrel/v1.md"): (1, 22),
    ("saffron", "seed/saffron/v1.md"): (1, 22),
    ("nimbus", "updates/nimbus/v2.md"): (2, 21),
}
META_FIELDS = {"company_id", "version", "doc_title", "section_id", "section_title", "chunk_id"}


@pytest.mark.parametrize(("company", "path"), list(EXPECTED))
def test_chunker_sections(company, path):
    version, n_sections = EXPECTED[(company, path)]
    policy = parse_policy(read_policy(path))
    chunks = chunk_policy(company, policy, max_words=150)

    assert policy.version == version
    assert [c.section_id for c in chunks] == [str(i) for i in range(1, n_sections + 1)]
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    for c in chunks:
        assert META_FIELDS <= c.metadata().keys()
        assert all(c.metadata()[f] not in ("", None) for f in META_FIELDS)
        assert c.chunk_id == f"{company}_v{version}_s{c.section_id}_p0"
        header = (
            f"Company: {policy.company_name} | {policy.doc_title} v{version}"
            f" | §{c.section_id} {c.section_title}"
        )
        assert c.text.startswith(header + "\n\n")
        assert c.text.endswith(c.body)


def test_chunker_is_deterministic():
    doc = read_policy("seed/kestrel/v1.md")
    a = chunk_policy("kestrel", parse_policy(doc), 150)
    b = chunk_policy("kestrel", parse_policy(doc), 150)
    assert a == b


def test_long_section_split_by_paragraph_keeps_heading():
    paras = "\n\n".join(" ".join([f"w{p}"] * 40) for p in range(5))  # 5 paras x 40 words
    doc = (
        "# Nimbus Software — Test Policy (Version 1, effective 2026-01-01)\n"
        f"## 1. Long Section\n{paras}\n## 2. Short\nTiny body.\n"
    )
    chunks = chunk_policy("nimbus", parse_policy(doc), max_words=100)
    long_parts = [c for c in chunks if c.section_id == "1"]
    assert [c.chunk_id for c in long_parts] == [f"nimbus_v1_s1_p{i}" for i in range(3)]
    for c in long_parts:
        assert c.text.startswith("Company: Nimbus Software | Test Policy v1 | §1 Long Section")
        assert len(c.body.split()) <= 100
    # no words lost
    assert sum(len(c.body.split()) for c in long_parts) == 200


def test_subsections_become_separate_chunks():
    doc = (
        "# Nimbus Software — Test Policy (Version 1, effective 2026-01-01)\n"
        "## 4. Night Rules\n### 4.1 Drops\nA.\n### 4.2 Night-shift drops\nB.\n"
    )
    chunks = chunk_policy("nimbus", parse_policy(doc), 150)
    assert [(c.section_id, c.section_title) for c in chunks] == [
        ("4.1", "Drops"),
        ("4.2", "Night-shift drops"),
    ]


@pytest.mark.parametrize(
    "doc",
    [
        "",
        "no title here\n## 1. A\nx",
        "# Nimbus Software — T (Version 1, effective 2026-01-01)\nstray text\n## 1. A\nx",
        "# Nimbus Software — T (Version 1, effective 2026-01-01)\n## 1. A\nx\n## 1. B\ny",
        "# Nimbus Software — T (Version 1, effective 2026-01-01)\nnothing",
    ],
    ids=["empty", "no-title", "text-before-section", "duplicate-id", "no-sections"],
)
def test_malformed_documents_rejected(doc):
    with pytest.raises(ChunkingError):
        parse_policy(doc)
