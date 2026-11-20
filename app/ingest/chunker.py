"""Split a policy document into section-level chunks (one chunk = one citable section)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    section_id: str      # e.g. "4.2"
    section_title: str   # e.g. "Night Shift Drops"
    text: str


def chunk_by_section(markdown: str) -> list[Section]:
    raise NotImplementedError  # TODO(D7): agree heading convention first
