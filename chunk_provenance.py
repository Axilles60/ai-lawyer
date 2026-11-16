"""Source locations for chunks, separate from citation and embedding text (03a §2.1.1)."""
from __future__ import annotations

from bisect import bisect_right
import json
import re

from kb_types import DocCard, Node

_HEADING = re.compile(r"^\s*(Глава|Раздел)\s+[\dIVXLCDMА-ЯЁ]+(?:[.\-]\d+)*\b.*$", re.I)
_REVISION = re.compile(r"\b(?:изменен\w*|изменён\w*|дополнен\w*|утратил\w*)\b", re.I)


def _normalized(text: str, start: int, end: int) -> tuple[str, list[int]]:
    words, offsets = [], []
    for match in re.finditer(r"\S+", text[start:end]):
        if words:
            words.append(" ")
            offsets.append(start + match.start() - 1)
        words.append(match.group())
        offsets.extend(range(start + match.start(), start + match.end()))
    return "".join(words), offsets


def _document(card: DocCard) -> str:
    name = card.act or card.title or card.file
    if card.title and card.title != name:
        name += " — " + card.title
    if card.number and card.number not in name:
        name += " № " + card.number
    if card.date:
        name += " (дата документа: " + card.date + ")"
    return name


def attach_provenance(card: DocCard, text: str, nodes: list[Node]) -> None:
    """Enrich every node; never alter source text, offsets, identifiers, or tree shape."""
    by_id = {node.node_id: node for node in nodes}
    newlines = [m.start() for m in re.finditer("\n", text)]
    headings = []
    position = 0
    chapter = section = ""
    for line in text.splitlines(keepends=True):
        match = _HEADING.match(line.rstrip("\r\n"))
        if match and not _REVISION.search(line):
            if match.group(1).lower() == "раздел":
                section, chapter = line.strip(), ""
            else:
                chapter = line.strip()
            headings.append((position, chapter, section))
        position += len(line)
    heading_starts = [row[0] for row in headings]
    fragments = {}
    cursors = {}
    for node in nodes:
        chain, seen, parent = [], set(), node
        while parent is not None and parent.node_id not in seen:
            seen.add(parent.node_id)
            description = parent.label or parent.path
            if parent.heading:
                description += " — " + parent.heading
            chain.append(description)
            parent = by_id.get(parent.parent_id)
        chain.reverse()
        start, end, scope = node.start, node.end, "node_range"
        if "fragment" in node.flags:
            parent = by_id.get(node.parent_id)
            scope = "parent_range"
            if parent is not None:
                start, end = parent.start, parent.end
                if 0 <= start < end <= len(text):
                    if parent.node_id not in fragments:
                        fragments[parent.node_id] = _normalized(text, start, end)
                    normalized, offsets = fragments[parent.node_id]
                    needle = " ".join(node.text.split())
                    found = normalized.find(needle, cursors.get(parent.node_id, 0)) if needle else -1
                    if found >= 0:
                        start, end = offsets[found], offsets[found + len(needle) - 1] + 1
                        cursors[parent.node_id] = found + 1  # Permit overlap, advance repeated passages.
                        scope = "fragment_range"
            else:
                start = end = 0
        valid = 0 <= start < end <= len(text)
        heading_index = bisect_right(heading_starts, start) - 1 if valid else -1
        chapter, section = headings[heading_index][1:] if heading_index >= 0 else ("", "")
        if not section:
            section = next((part for part in chain if part.startswith("раздел ")), "")
        part = re.search(r"(?:^|/)ч\.(\d+(?:\.\d+)*)", node.path)
        node.provenance = {
            "document": _document(card), "file": card.file, "doc_id": card.doc_id,
            "node_id": node.node_id, "document_sha256": card.sha256,
            "chapter": chapter, "section": section, "part": part.group(1) if part else "",
            "hierarchy": chain, "path": node.path, "norm": node.label,
            "line_start": bisect_right(newlines, start) + 1 if valid else None,
            "line_end": bisect_right(newlines, end - 1) + 1 if valid else None,
            "line_basis": "normalized_source", "range_scope": scope if valid else "unknown",
        }


def source_header(node: Node, card: DocCard, document_tag: str = "") -> str:
    """Compact, escaped source data for a node shown to the LLM, including adapters."""
    meta = getattr(node, "provenance", None) or {}
    first, last = meta.get("line_start"), meta.get("line_end")
    scope = meta.get("range_scope", "unknown")
    lines = f"{first}–{last}" if first is not None and last is not None else "не определены"
    area = {"node_range": "диапазон узла", "fragment_range": "фрагмент",
            "parent_range": "диапазон родителя", "unknown": "не определена"}.get(scope, "не определена")
    if document_tag:
        # Full document identity/file is declared once in its block, keeping context useful.
        values = [("документ", document_tag), ("глава", meta.get("chapter") or "—"),
                  ("раздел", meta.get("section") or "—"), ("часть", meta.get("part") or "—"),
                  ("путь", meta.get("path") or node.path), ("строки", lines)]
        if scope in ("parent_range", "unknown"):
            values.append(("область", area))
        return "Источник чанка: " + "; ".join(k + "=" + json.dumps(v, ensure_ascii=False) for k, v in values)
    data = {"документ": meta.get("document") or _document(card),
            "файл": meta.get("file") or card.file,
            "глава": meta.get("chapter") or "не определена",
            "раздел": meta.get("section") or "не определён",
            "часть": meta.get("part") or "не определена",
            "иерархия": meta.get("hierarchy") or [node.label or node.path],
            "норма": meta.get("norm") or node.label or node.path,
            "строки нормализованного исходника": lines, "область": area}
    return "Источник чанка: " + json.dumps(data, ensure_ascii=False, separators=(",", ":"))
