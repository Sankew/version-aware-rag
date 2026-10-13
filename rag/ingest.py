"""Read the manifest, parse md/HTML/PDF into sections, and pack sections into chunks."""
import hashlib
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
CHUNK_TOKENS, OVERLAP_TOKENS = 400, 60  # bge-small truncates at 512; 60 is ~15% overlap


@dataclass(frozen=True)
class Section:
    heading: str
    text: str


@dataclass(frozen=True)
class Chunk:
    document_id: str
    version: int
    section: str
    ord: int
    content: str


def manifest(path: Path = ROOT / "corpus" / "manifest.jsonl") -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse(path: Path, title: str) -> list[Section]:
    """Split a document into (heading, text) sections. PDFs become one section named after the title."""
    if path.suffix == ".md":
        return _sections(path.read_text(encoding="utf-8").splitlines(), lambda line: re.match(r"#{1,6}\s+(.*)", line))
    if path.suffix == ".html":
        parser = _HTML()
        parser.feed(path.read_text(encoding="utf-8"))
        parser.close()
        parser.end_row()  # a row or text left open at the end of the page
        parser.flush()
        return _sections(parser.lines, lambda line: re.match(r"\x00(.*)", line))
    if path.suffix == ".pdf":
        from pypdf import PdfReader
        text = "\n".join(page.extract_text() for page in PdfReader(path).pages)
        return [Section(title, _clean(text))]
    raise ValueError(f"unsupported file type: {path}")


def _sections(lines: list[str], heading: Callable[[str], re.Match | None]) -> list[Section]:
    sections: list[tuple[str, list[str]]] = []
    preamble: list[str] = []
    for line in lines:
        match = heading(line)
        if match:
            sections.append((match.group(1).strip(), []))
        elif line.strip():
            (sections[-1][1] if sections else preamble).append(line.strip())
    if not sections:
        return [Section("", _clean("\n".join(preamble)))]
    first, body = sections[0]
    sections[0] = (first, preamble + body)  # navigation and banners above the first heading
    return [Section(h, _clean("\n".join(body))) for h, body in sections if body or h]


def _clean(text: str) -> str:
    return re.sub(r"\*\*|__", "", text).strip()


class _HTML(HTMLParser):
    """Headings become lines prefixed with NUL; table rows become ' | '-joined lines.
    Rows are closed by </tr>, the next <tr>, </table> or the end of the page, since HTML lets </tr> be omitted."""

    BLOCKS = {"p", "li", "div", "tr", "br", "header", "footer", "table", "ul", "h1", "h2", "h3", "h4"}

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []
        self.buffer: list[str] = []
        self.skip = 0
        self.cells: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("script", "style", "title"):
            self.skip += 1
        elif tag == "tr":
            self.end_row()
            self.flush()
            self.cells = []
        elif tag in ("td", "th"):
            self.flush_cell()
        elif tag in self.BLOCKS:
            self.flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "title"):
            self.skip -= 1
        elif tag in ("h1", "h2", "h3", "h4"):
            self.lines.append("\x00" + self.take())
        elif tag in ("td", "th"):
            self.flush_cell()
        elif tag in ("tr", "table"):
            self.end_row()
            self.flush()
        elif tag in self.BLOCKS:
            self.flush()

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.buffer.append(data)

    def take(self) -> str:
        text = " ".join("".join(self.buffer).split())
        self.buffer = []
        return text

    def flush_cell(self) -> None:
        text = self.take()
        if self.cells is not None and text:
            self.cells.append(text)

    def end_row(self) -> None:
        if self.cells is not None:
            self.flush_cell()
            self.lines.append(" | ".join(self.cells))
            self.cells = None

    def flush(self) -> None:
        if self.cells is not None:
            self.flush_cell()
        elif text := self.take():
            self.lines.append(text)


def words(text: str) -> int:
    """Rough token count for when the bge tokenizer is not installed."""
    return len(re.findall(r"\w+|[^\w\s]", text))


def chunk(row: dict, sections: list[Section], count: Callable[[str], int] = words,
          budget: int = CHUNK_TOKENS, overlap: int = OVERLAP_TOKENS) -> list[Chunk]:
    """Pack each section's lines into chunks under `budget` tokens, carrying ~`overlap` tokens forward.

    Every chunk starts with "title, section" so a retrieved chunk keeps its qualifiers
    (whose policy, which scope). Sections with no text are skipped. Carried lines are dropped
    when they would push the next chunk over budget. A single line longer than the budget is
    kept whole; split such lines by sentence if a corpus ever has them.
    """
    chunks: list[Chunk] = []
    for section in sections:
        if not section.text.strip():
            continue
        header = f"{row['title']}, {section.heading}" if section.heading else row["title"]
        window: list[str] = []
        for line in section.text.splitlines():
            if window and count("\n".join([header, *window, line])) > budget:
                chunks.append(_make(row, section.heading, len(chunks), header, window))
                carried: list[str] = []
                while window and count("\n".join(window[-1:] + carried)) <= overlap:
                    carried.insert(0, window.pop())
                while carried and count("\n".join([header, *carried, line])) > budget:
                    carried.pop(0)
                window = carried
            window.append(line)
        chunks.append(_make(row, section.heading, len(chunks), header, window))
    return chunks


def _make(row: dict, heading: str, ord: int, header: str, lines: list[str]) -> Chunk:
    return Chunk(row["document_id"], row["version"], heading or row["title"], ord, "\n".join([header, *lines]).strip())


def load_corpus(count: Callable[[str], int] = words) -> list[tuple[dict, list[Chunk]]]:
    """(manifest row, chunks) per file, in upload order, the order documents reached the system."""
    rows = sorted(manifest(), key=lambda r: (r["uploaded_at"], r["document_id"], r["version"]))
    return [(row, chunk(row, parse(ROOT / row["path"], row["title"]), count)) for row in rows]


def fingerprint() -> str:
    """Hash of the manifest, every corpus file and this module's code: changes when the chunks would."""
    digest = hashlib.sha256(Path(__file__).read_bytes())
    for row in manifest():
        digest.update(json.dumps(row, sort_keys=True).encode())
        digest.update((ROOT / row["path"]).read_bytes())
    return digest.hexdigest()[:16]
