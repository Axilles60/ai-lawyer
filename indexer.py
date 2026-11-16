"""Индекс базы знаний (03a; 06-implementation-guide §5).

T1.4 — доступ к документам и узлам, адаптивный родитель, малые документы целиком, шапки, опись базы.
T1.5 — хранилище SQLite (документы, узлы, связи, метаданные индекса) и полнотекстовый поиск FTS5 с
собственной нормализацией (textnorm); если FTS5 в сборке SQLite нет — тот же поиск через bm25s.
Векторы (T1.8) добавляются следующей задачей.

Index после сборки только читается — безопасен для потоков (запросы к SQLite — под блокировкой).
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import tempfile
import threading
import time
import weakref
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz

from emb_cache import EmbeddingCache, text_key, usable_vector, stored_vector
from kb_parser import build_entities, parse_all
from llm_client import LLMError, log
from kb_types import DocCard, Entity, EntityMatch, Node, ParsedDoc, RawDoc, Relation
from textnorm import doc_number_token, index_tokens, special_tokens, tokens

OWN_PRONOUNS_RE = re.compile(r"(?<![а-яё])(мы|нас|нам|нами|наш\w*)(?![а-яё])", re.IGNORECASE)
FUZZY_MIN_LEN = 5      # нечёткое сравнение основ только для слов не короче
FUZZY_RATIO = 88       # rapidfuzz.ratio для опечаток в названиях


def _stem_eq(a: str, b: str) -> bool:
    if a == b:
        return True
    return min(len(a), len(b)) >= FUZZY_MIN_LEN and fuzz.ratio(a, b) >= FUZZY_RATIO


def _find_seq(haystack: list[str], needle: tuple[str, ...]) -> int:
    """Позиция первого вхождения последовательности основ (с учётом опечаток) или -1."""
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if all(_stem_eq(haystack[i + j], needle[j]) for j in range(n)):
            return i
    return -1


def _contains_seq(haystack: list[str], needle: tuple[str, ...]) -> bool:
    return _find_seq(haystack, needle) >= 0


def _same_org(name: str, ent: Entity) -> bool:
    return name.lower().replace("ё", "е") in {a.lower().replace("ё", "е") for a in ent.aliases} or \
        name.lower().replace("ё", "е") == ent.canonical.lower().replace("ё", "е")

SCHEMA_VERSION = "slice-1"
ROOT_DIR = Path(__file__).resolve().parent


def _cleanup_db(conn: sqlite3.Connection, path: str) -> None:
    try:
        conn.close()
    except sqlite3.Error:
        pass
    try:
        os.remove(path)
    except OSError:
        pass

KIND_TITLES = {
    "contract": "Договоры",
    "amendment": "Дополнительные соглашения",
    "regulation": "Внутренние регламенты и положения",
    "letter": "Письма",
    "memo": "Служебные записки",
    "other": "Прочие документы",
}
KIND_ORDER = ["contract", "amendment", "regulation", "letter", "memo", "other"]


def fmt_date(iso: str) -> str:
    """2024-08-01 → 01.08.2024."""
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", iso or "")
    return f"{m.group(3)}.{m.group(2)}.{m.group(1)}" if m else ""


def _period_start(date: str) -> str:
    """«2024-08» → «2024-08-01»; «2024» → «2024-01-01»; полная дата — как есть."""
    m = re.fullmatch(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", date or "")
    if not m:
        return ""
    return f"{m.group(1)}-{m.group(2) or '01'}-{m.group(3) or '01'}"


MONTHS_PREP = ["январе", "феврале", "марте", "апреле", "мае", "июне", "июле", "августе", "сентябре", "октябре",
               "ноябре", "декабре"]


def _fmt_period(date: str) -> str:
    """«2024-08» → «в августе 2024 г.»; «2024» → «в 2024 г.»."""
    m = re.fullmatch(r"(\d{4})(?:-(\d{2}))?", date or "")
    if not m:
        return f"на {date}"
    if m.group(2):
        return f"в {MONTHS_PREP[int(m.group(2)) - 1]} {m.group(1)} г."
    return f"в {m.group(1)} г."


def plural_docs(n: int) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} документ"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return f"{n} документа"
    return f"{n} документов"


def _article_key(number: str) -> tuple:
    return tuple(int(x) if x.isdigit() else 0 for x in re.split(r"[.\-]", number or "0"))


def doc_title(card: DocCard) -> str:
    """Краткое наименование документа для ссылок и описи: «Договор поставки № П-17/2024 от 12.02.2024»."""
    if card.kind == "law":
        return f"{card.act}, ст. {card.number}" if card.number else card.act or card.title
    parts = [card.act or card.title or card.file]
    if card.number:
        parts.append(f"№ {card.number}")
    if card.kind == "regulation" and card.date:
        parts.append(f"(ред. от {fmt_date(card.date)})")
    elif card.date:
        parts.append(f"от {fmt_date(card.date)}")
    return " ".join(parts)


def status_text(card: DocCard) -> str:
    if card.status == "утратил силу":
        return f"утратил силу с {fmt_date(card.valid_to)}" if card.valid_to else "утратил силу"
    if card.valid_from and card.valid_from != card.date:
        return f"действует с {fmt_date(card.valid_from)}"
    return card.status or "действует"


def doc_header(card: DocCard) -> str:
    """Шапка документа для контекста генерации (03a §2.2): вид, номер, дата, стороны, статус."""
    if card.kind == "law":
        title = f" «{card.title}»" if card.title else ""
        export = f", выгрузка {fmt_date(card.export_date)}" if card.export_date else ""
        return f"{doc_title(card)}{title} (действующая редакция{export})"
    pieces = [doc_title(card)]
    if card.kind in ("letter", "memo") and card.title and card.title != card.act:
        pieces.append(f"«{card.title}»")
    if card.parties:
        sides = ", ".join(f"{name} ({role})" if role else name for name, role, _ in card.parties)
        pieces.append(("издатель: " if card.parties[0][1] == "издатель" else "стороны: ")
                      + sides.replace(" (издатель)", ""))
    pieces.append(status_text(card))
    return "; ".join(pieces)


@dataclass
class Index:
    parsed: list[ParsedDoc]
    relations: list[Relation]
    parent_max_chars: int = 3500
    whole_doc_chars: int = 8000
    inventory_max_chars: int = 3000
    use_safe_context: bool = False
    safe_document_spans: dict = field(default_factory=dict, repr=False)
    _cards: dict[str, DocCard] = field(default_factory=dict, repr=False)
    _nodes: dict[str, Node] = field(default_factory=dict, repr=False)
    _children: dict[str | None, list[str]] = field(default_factory=dict, repr=False)
    _doc_roots: dict[str, list[str]] = field(default_factory=dict, repr=False)
    _subtree_len: dict[str, int] = field(default_factory=dict, repr=False)
    _inventory: str = field(default="", repr=False)
    use_fts5: bool | None = None          # None — определить автоматически; False — принудительно bm25s
    db_path: str = field(default="", repr=False)
    fts_backend: str = field(default="", repr=False)   # fts5 | bm25s
    _db: sqlite3.Connection | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _search_ids: list[str] = field(default_factory=list, repr=False)
    _bm25: object = field(default=None, repr=False)
    entities: list[Entity] = field(default_factory=list, repr=False)
    _aliases: list[tuple[Entity, str, tuple[str, ...], int, str]] = field(default_factory=list, repr=False)
    _mentions: dict[str, list[tuple[str, str]]] = field(default_factory=dict, repr=False)
    # Векторы текстов узлов (T1.8): строки выровнены с vec_ids; has_vec — у каких строк вектор есть
    vec_ids: list[str] = field(default_factory=list, repr=False)
    vectors: np.ndarray | None = field(default=None, repr=False)
    has_vec: np.ndarray | None = field(default=None, repr=False)
    emb_stats: dict = field(default_factory=dict, repr=False)
    build_seconds: float = 0.0

    def __post_init__(self) -> None:
        children: dict[str | None, list[str]] = defaultdict(list)
        for p in self.parsed:
            if self.use_safe_context:
                from safe_context import find_spans
                self.safe_document_spans[p.card.doc_id] = find_spans(p.text)
            self._cards[p.card.doc_id] = p.card
            roots = []
            for n in p.nodes:
                self._nodes[n.node_id] = n
                if n.parent_id:
                    children[n.parent_id].append(n.node_id)
                else:
                    roots.append(n.node_id)
            self._doc_roots[p.card.doc_id] = roots
        self._children = dict(children)
        for p in self.parsed:
            for n in reversed(p.nodes):  # дети идут после родителей
                self._subtree_len[n.node_id] = len(self.subtree_text(n.node_id))
            # Малый документ — в контекст целиком (03a §2.2): по размеру отображаемого текста, не по типу
            p.card.whole = len(self.doc_text(p.card.doc_id)) <= self.whole_doc_chars
        self._build_entities()
        self._inventory = self._build_inventory()
        self._build_store()

    # ---------------- сущности и алиасы (T1.6) ----------------

    def _build_entities(self) -> None:
        """Каталог сущностей, алиасы в нормализованной форме и упоминания организаций в узлах."""
        self.entities = build_entities(self.parsed)
        # Единое написание организаций в карточках (шапки, опись): каноническое имя из каталога
        for ent in self.entities:
            if ent.kind != "org":
                continue
            for doc_id in ent.doc_ids:
                card = self._cards[doc_id]
                card.parties = [(ent.canonical if _same_org(name, ent) else name, role, inn)
                                for name, role, inn in card.parties]
        for ent in self.entities:
            for alias in ent.aliases:
                stems = tuple(tokens(alias))
                # Алиас без содержательного слова (только цифры и однобуквенные части) по основам не сопоставляем:
                # «К-12» → «12» совпало бы с любым числом в вопросе
                # (у актов сокращения короткие — «ТК», «НК», «152-ФЗ»: для них достаточно двух букв)
                min_len = 2 if ent.kind == "act" else 3
                if not any(len(s) >= min_len and re.search(r"[a-zа-я]", s) for s in stems):
                    stems = ()
                number_tok = ""
                if ent.kind == "document":
                    card = self._cards[ent.doc_ids[0]]
                    if card.number and alias == card.number:
                        number_tok = doc_number_token(card.number)
                        stems = ()  # номер документа — только по спецтокену номера
                    # Вид документа («Договор поставки», «Письмо») — общее имя: слабее номера и названия
                    weight = 80 if alias in (card.act, card.title) else 100
                else:
                    weight = 100
                if stems or number_tok:
                    self._aliases.append((ent, alias, stems, weight, number_tok))
        orgs = [e for e in self.entities if e.kind == "org"]
        for node in self._nodes.values():
            node_stems = tokens(node.text)
            found = []
            for ent in orgs:
                for alias in ent.aliases:
                    stems = tuple(tokens(alias))
                    if stems and _contains_seq(node_stems, stems):
                        found.append((ent.entity_id, ent.roles.get(node.doc_id, "")))
                        break
            if found:
                self._mentions[node.node_id] = found

    # ---------------- прямой доступ по реквизитам (канал C0, T1.12) ----------------

    def _label_tokens(self, node: Node) -> frozenset:
        return frozenset(t for t in special_tokens(node.label) if not t.startswith("no"))

    def docs_for_ref(self, doc_ref: str) -> list[str]:
        """Документы, названные ссылкой: номер документа, вид, организация, акт («ТК», «152-ФЗ»)."""
        if not doc_ref:
            return []
        ids: list[str] = []
        numbers = {t for t in special_tokens(doc_ref) if t.startswith("no")}
        for c in self._cards.values():
            if c.number and doc_number_token(c.number) in numbers:
                ids.append(c.doc_id)
        for m in self.resolve(doc_ref):
            if m.full and not (self.own_org() and m.entity_id == self.own_org().entity_id):
                ids.extend(d for d in m.doc_ids if d not in ids)
        return ids

    def lookup(self, doc_ref: str, node_ref: str) -> list[str]:
        """C0 (03b §2): узлы по реквизитам. node_ref — «ст. 70», «ч. 2», «п. 3.2», «ст. 59, часть первая»;
        doc_ref — номер документа, вид, контрагент или акт (пусто — по всей базе).
        Возвращает узлы, чья ссылка точно совпадает с названным положением (сначала верхние уровни);
        если точного совпадения нет — ближайшие более конкретные."""
        wanted = frozenset(t for t in special_tokens(node_ref) if not t.startswith("no"))
        if not wanted:
            return []
        docs = set(self.docs_for_ref(doc_ref)) if doc_ref else set(self._cards)
        matches = [n for n in self._nodes.values()
                   if n.doc_id in docs and "fragment" not in n.flags and wanted <= self._label_tokens(n)]
        if not matches:
            return []
        best = min(len(self._label_tokens(n)) for n in matches)
        chosen = [n for n in matches if len(self._label_tokens(n)) == best]
        chosen.sort(key=lambda n: (n.level, n.start))
        return [n.node_id for n in chosen]

    def entity(self, entity_id: str) -> Entity:
        return next(e for e in self.entities if e.entity_id == entity_id)

    def own_org(self) -> Entity | None:
        return next((e for e in self.entities if e.own), None)

    def mentions(self, node_id: str) -> list[tuple[str, str]]:
        """Организации, упомянутые в узле: (entity_id, роль в документе узла)."""
        return self._mentions.get(node_id, [])

    def resolve(self, text: str) -> list[EntityMatch]:
        """Организации, документы и акты, названные в тексте (03b §1.2). По убыванию уверенности:
        100 — номер документа или полное название; 80 — вид документа («договор поставки»);
        60 — начало длинного названия; 50 — первое слово названия организации («Альфа»);
        40 — «мы/наш» → своя организация. Неоднозначность (несколько частичных совпадений) решает вызывающий."""
        q_stems = tokens(text)
        q_special = set(special_tokens(text))
        best: dict[str, EntityMatch] = {}
        spans: dict[str, tuple[int, int]] = {}  # entity_id -> (начало, конец) полного совпадения в запросе

        def offer(ent: Entity, score: int, matched: str, full: bool, span: tuple[int, int] | None = None) -> None:
            cur = best.get(ent.entity_id)
            if cur is None or score > cur.score or (score == cur.score and span and ent.entity_id in spans
                                                    and span[1] - span[0] > spans[ent.entity_id][1]
                                                    - spans[ent.entity_id][0]):
                best[ent.entity_id] = EntityMatch(ent.entity_id, ent.kind, ent.canonical, tuple(ent.doc_ids),
                                                  score, matched, full)
                if span:
                    spans[ent.entity_id] = span

        for ent, alias, stems, weight, number_tok in self._aliases:
            if number_tok and number_tok in q_special:
                offer(ent, 100, alias, True)
                continue
            if not stems:
                continue
            pos = _find_seq(q_stems, stems)
            if pos >= 0:
                offer(ent, weight, alias, True, (pos, pos + len(stems)))
            elif len(stems) >= 3 and _contains_seq(q_stems, stems[:2]):
                offer(ent, 60, alias, False)
            elif ent.kind == "org" and len(stems) >= 2 and len(stems[0]) >= 4 and any(
                    _stem_eq(q, stems[0]) for q in q_stems):
                offer(ent, 50, alias.split()[0] if " " in alias else stems[0], False)
        # Самое длинное совпадение побеждает: «Ромашка» внутри найденного «Ромашка Плюс» — не отдельная сущность
        for eid, (start, end) in list(spans.items()):
            if any(o != eid and o_start <= start and end <= o_end and (o_end - o_start) > (end - start)
                   and best[o].kind == best[eid].kind for o, (o_start, o_end) in spans.items()):
                m = best[eid]
                best[eid] = EntityMatch(m.entity_id, m.kind, m.canonical, m.doc_ids, 50, m.matched, False)
        own = self.own_org()
        if own and OWN_PRONOUNS_RE.search(text):
            offer(own, 40, OWN_PRONOUNS_RE.search(text).group(0), False)
        return sorted(best.values(), key=lambda m: (-m.score, m.kind, m.canonical))

    # ---------------- хранилище и полнотекстовый поиск (T1.5) ----------------

    def searchable_nodes(self) -> list[Node]:
        """Узлы, по которым ищем: всё, кроме раздробленных родителей (вместо них — их фрагменты)."""
        return [n for n in self._nodes.values() if "split" not in n.flags and n.text.strip()]

    def search_text(self, node_id: str) -> str:
        """Путь + вводная фраза + текст: то, что идёт в FTS, эмбеддинг и реранк (06 §5)."""
        node = self._nodes[node_id]
        card = self._cards[node.doc_id]
        path = " | ".join(x for x in (doc_title(card), node.label, node.heading) if x)
        content = node.text
        if node.contextual_text:
            content = f"[Контекст: {node.contextual_text}] " + content
        text = "\n".join(x for x in (path, node.lead_in, content) if x)
        if self.use_safe_context:
            from safe_context import clean
            return clean(text)
        return text

    def _open_db(self) -> sqlite3.Connection:
        """Файл во временном каталоге (рабочий каталог платформы может быть только для чтения); запасной — :memory:."""
        try:
            fd, path = tempfile.mkstemp(prefix="kb_index_", suffix=".sqlite")
            os.close(fd)
            conn = sqlite3.connect(path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=OFF")
            self.db_path = path
            weakref.finalize(self, _cleanup_db, conn, path)  # временный файл удаляется вместе с индексом
            return conn
        except (OSError, sqlite3.Error):
            self.db_path = ":memory:"
            return sqlite3.connect(":memory:", check_same_thread=False)

    def _build_store(self) -> None:
        db = self._open_db()
        db.executescript("""
            CREATE TABLE documents (doc_id TEXT PRIMARY KEY, file TEXT, sha256 TEXT, kind TEXT, act TEXT,
                number TEXT, date TEXT, title TEXT, parties TEXT, status TEXT, valid_from TEXT, valid_to TEXT,
                condition TEXT, source_url TEXT, export_date TEXT, chars INTEGER, whole INTEGER);
            CREATE TABLE nodes (node_id TEXT PRIMARY KEY, doc_id TEXT, parent_id TEXT, level INTEGER, path TEXT,
                label TEXT, heading TEXT, text TEXT, lead_in TEXT, start INTEGER, "end" INTEGER, flags TEXT,
                notes TEXT, provenance TEXT, contextual_text TEXT);
            CREATE INDEX nodes_doc ON nodes(doc_id);
            CREATE TABLE relations (src_doc TEXT, dst_doc TEXT, kind TEXT, date TEXT, condition TEXT);
            CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE entities (entity_id TEXT PRIMARY KEY, kind TEXT, canonical TEXT, inn TEXT, own INTEGER,
                doc_ids TEXT, roles TEXT);
            CREATE TABLE aliases (entity_id TEXT, alias TEXT);
            CREATE TABLE mentions (node_id TEXT, entity_id TEXT, role TEXT);
        """)
        db.executemany("INSERT INTO entities VALUES (?,?,?,?,?,?,?)", [
            (e.entity_id, e.kind, e.canonical, e.inn, int(e.own), json.dumps(e.doc_ids),
             json.dumps(e.roles, ensure_ascii=False)) for e in self.entities])
        db.executemany("INSERT INTO aliases VALUES (?,?)", [(e.entity_id, a) for e in self.entities for a in e.aliases])
        db.executemany("INSERT INTO mentions VALUES (?,?,?)",
                       [(nid, eid, role) for nid, items in self._mentions.items() for eid, role in items])
        db.executemany("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
            (c.doc_id, c.file, c.sha256, c.kind, c.act, c.number, c.date, c.title,
             json.dumps(c.parties, ensure_ascii=False), c.status, c.valid_from, c.valid_to, c.condition,
             c.source_url, c.export_date, c.chars, int(c.whole)) for c in self._cards.values()])
        db.executemany("INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
            (n.node_id, n.doc_id, n.parent_id, n.level, n.path, n.label, n.heading, n.text, n.lead_in, n.start,
             n.end, json.dumps(n.flags), json.dumps(n.notes, ensure_ascii=False), json.dumps(n.provenance, ensure_ascii=False),
             n.contextual_text) for n in self._nodes.values()])
        db.executemany("INSERT INTO relations VALUES (?,?,?,?,?)",
                       [(r.src_doc, r.dst_doc, r.kind, r.date, r.condition) for r in self.relations])
        db.executemany("INSERT INTO index_meta VALUES (?,?)", [
            ("schema_version", SCHEMA_VERSION),
            ("doc_hashes", json.dumps({c.doc_id: c.sha256 for c in self._cards.values()})),
            ("params", json.dumps({"parent_max_chars": self.parent_max_chars, "whole_doc_chars": self.whole_doc_chars,
                                   "inventory_max_chars": self.inventory_max_chars})),
        ])

        searchable = self.searchable_nodes()
        self._search_ids = [n.node_id for n in searchable]
        token_lists = [index_tokens(self.search_text(n.node_id)) for n in searchable]
        backend = "fts5"
        if self.use_fts5 is False:
            backend = "bm25s"
        else:
            try:
                db.execute("CREATE VIRTUAL TABLE nodes_fts USING fts5(node_id UNINDEXED, tokens, "
                           "tokenize='unicode61 remove_diacritics 0')")
                db.executemany("INSERT INTO nodes_fts VALUES (?, ?)",
                               [(nid, " ".join(toks)) for nid, toks in zip(self._search_ids, token_lists)])
            except sqlite3.OperationalError as exc:
                print(f"[indexer] FTS5 недоступен ({exc}); полнотекстовый поиск через bm25s", file=sys.stderr)
                backend = "bm25s"
        if backend == "bm25s":
            import bm25s  # из разрешённого requirements.txt; нужен только без FTS5
            retriever = bm25s.BM25()
            retriever.index([toks or ["_"] for toks in token_lists], show_progress=False)
            self._bm25 = retriever
        db.execute("INSERT INTO index_meta VALUES ('fts_backend', ?)", (backend,))
        db.commit()
        self.fts_backend = backend
        self._db = db

    def search_fts(self, query: str | list[str], k: int = 40) -> list[tuple[str, float]]:
        """Полнотекстовый поиск: (node_id, score) по убыванию; score больше — лучше.
        query — текст запроса или готовые токены textnorm.index_tokens."""
        toks = index_tokens(query) if isinstance(query, str) else list(query)
        toks = list(dict.fromkeys(t for t in toks if t))
        if not toks or not self._search_ids:
            return []
        if self.fts_backend == "fts5":
            match = " OR ".join('"' + t.replace('"', '""') + '"' for t in toks)
            with self._lock:
                rows = self._db.execute(
                    "SELECT node_id, bm25(nodes_fts) AS s FROM nodes_fts WHERE nodes_fts MATCH ? ORDER BY s LIMIT ?",
                    (match, k)).fetchall()
            return [(nid, -float(s)) for nid, s in rows]
        scores = np.asarray(self._bm25.get_scores(toks), dtype=np.float32)
        order = np.argsort(-scores)[:k]
        return [(self._search_ids[i], float(scores[i])) for i in order if scores[i] > 0]

    def search_dense(self, vector: np.ndarray, k: int = 40) -> list[tuple[str, float]]:
        """Поиск по векторам текстов узлов: (node_id, косинус) по убыванию. Без векторов — пусто (C2)."""
        if self.vectors is None or vector is None or not len(self.vec_ids):
            return []
        q = np.asarray(vector, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.vectors.shape[1]:
            return []
        norm = float(np.linalg.norm(q)) or 1.0
        scores = self.vectors.astype(np.float32) @ (q / norm)
        scores[~self.has_vec] = -np.inf
        k = min(k, int(self.has_vec.sum()))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.vec_ids[i], float(scores[i])) for i in top]

    def report(self) -> dict:
        """Отчёт индексации (03a §5): документы, узлы, предупреждения разбора, сущности, связи, поиск, векторы."""
        kinds = Counter(c.kind for c in self._cards.values())
        levels = Counter(n.level for n in self._nodes.values())
        flags = Counter(f for n in self._nodes.values() for f in n.flags)
        warnings = {p.card.file: p.warnings for p in self.parsed if p.warnings}
        searchable = self.searchable_nodes()
        lengths = sorted(len(n.text) for n in searchable)
        return {
            "documents": len(self._cards), "by_kind": dict(kinds),
            "whole_documents": sum(c.whole for c in self._cards.values()),
            "nodes": len(self._nodes), "by_level": dict(sorted(levels.items())),
            "searchable_nodes": len(searchable),
            "node_chars_median": lengths[len(lengths) // 2] if lengths else 0,
            "node_chars_max": lengths[-1] if lengths else 0,
            "flags": dict(flags), "parse_warnings": warnings,
            "relations": Counter(r.kind for r in self.relations),
            "entities": dict(Counter(e.kind for e in self.entities)),
            "own_org": self.own_org().canonical if self.own_org() else "",
            "mentions": sum(len(v) for v in self._mentions.values()),
            "fts_backend": self.fts_backend, "inventory_chars": len(self._inventory),
            "embeddings": self.emb_stats or {"with_vector": 0},
            "build_seconds": self.build_seconds,
        }

    def report_text(self) -> str:
        r = self.report()
        emb = r["embeddings"]
        lines = [
            f"Индекс: документов {r['documents']} {r['by_kind']}, целиком {r['whole_documents']}; "
            f"узлов {r['nodes']} (поиск по {r['searchable_nodes']}, медиана {r['node_chars_median']} симв., "
            f"макс. {r['node_chars_max']}); связи {dict(r['relations'])}; сущности {r['entities']}; "
            f"упоминаний {r['mentions']}; FTS: {r['fts_backend']}",
            f"Векторы: {emb.get('with_vector', 0)}/{r['searchable_nodes']} (кэш {emb.get('cache_hits', 0)}, "
            f"сбойных батчей {emb.get('failed_batches', 0)}, пропущено по дедлайну "
            f"{emb.get('skipped_by_deadline', 0)}, {emb.get('seconds', 0)} с); сборка {r['build_seconds']} с",
        ]
        shipped = emb.get("shipped") or {}
        if shipped.get("status", "none") != "none":
            lines.append("Готовые векторы ZIP: " + ", ".join(f"{k} {v}" for k, v in shipped.items()))
        for file, warns in r["parse_warnings"].items():
            lines.append(f"Предупреждение разбора [{file}]: {'; '.join(warns)}")
        return "\n".join(lines)

    def sql(self, query: str, params: tuple = ()) -> list[tuple]:
        """Чтение из хранилища (отладка, отчёт индексации)."""
        with self._lock:
            return self._db.execute(query, params).fetchall()

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None
        if self.db_path and self.db_path != ":memory:":
            try:
                os.remove(self.db_path)
            except OSError:
                pass

    # ---------------- доступ ----------------

    def docs(self) -> list[DocCard]:
        return list(self._cards.values())

    def doc(self, doc_id: str) -> DocCard:
        return self._cards[doc_id]

    def node(self, node_id: str) -> Node:
        return self._nodes[node_id]

    def nodes(self) -> list[Node]:
        return list(self._nodes.values())

    def children(self, node_id: str) -> list[Node]:
        return [self._nodes[c] for c in self._children.get(node_id, [])]

    def doc_nodes(self, doc_id: str) -> list[Node]:
        return [n for n in self._nodes.values() if n.doc_id == doc_id]

    def status_on(self, doc_id: str, date: str | None = None) -> str:
        """Статус документа на дату события (03b §1.4, P5.7): готовый факт для генератора.
        date — ISO «2024-08-01», «2024-08» или «2024»; без даты — статус на сегодня (действующая редакция, Q13).
        Законы — только действующие редакции: дата события на их статус не влияет (Q1)."""
        card = self._cards[doc_id]
        if card.kind == "law":
            return "действующая редакция"
        cond = f"; условие: {card.condition}" if card.condition else ""
        if not date:
            return status_text(card) + cond
        day = _period_start(date)
        if not day:
            return status_text(card) + cond
        shown = f"на {fmt_date(day)}" if len(date) >= 10 else _fmt_period(date)
        if card.valid_from and day < card.valid_from:
            return f"не действовал {shown}: действует с {fmt_date(card.valid_from)}{cond}"
        if card.valid_to and day >= card.valid_to:
            return f"не действовал {shown}: утратил силу с {fmt_date(card.valid_to)}{cond}"
        return f"действовал {shown}{cond}"

    def related_docs(self, doc_id: str) -> list[Relation]:
        """Связи документа в обе стороны: доп. соглашения, другие редакции, письма-дополнения."""
        return [r for r in self.relations if doc_id in (r.src_doc, r.dst_doc)]

    # ---------------- текст ----------------

    def subtree_text(self, node_id: str) -> str:
        """Текст узла вместе с потомками в порядке документа. У раздробленного узла — его полный текст,
        фрагменты не дублируются."""
        node = self._nodes[node_id]
        parts = [node.text] if node.text else []
        for child_id in self._children.get(node_id, []):
            child = self._nodes[child_id]
            if "fragment" in child.flags:
                continue
            parts.append(self.subtree_text(child_id))
        return "\n".join(p for p in parts if p)

    def doc_text(self, doc_id: str) -> str:
        """Полный текст документа без шапки выгрузки и редакционных пометок — для «документа целиком»."""
        return "\n".join(self.subtree_text(r) for r in self._doc_roots.get(doc_id, []))

    def context_parent(self, node_id: str, allow_whole: bool = True) -> str:
        """Адаптивный родитель (03a §2.2): документ целиком, если он мал (и allow_whole); иначе самый верхний
        предок, чей текст с потомками укладывается в parent_max_chars; иначе сам узел."""
        node = self._nodes[node_id]
        if allow_whole and self._cards[node.doc_id].whole:
            return node.doc_id
        best = node
        while best.parent_id:
            parent = self._nodes[best.parent_id]
            if self._subtree_len[parent.node_id] > self.parent_max_chars:
                break
            best = parent
        return best.node_id

    def context_text(self, target: str) -> str:
        """Текст того, что вернул context_parent: документ целиком, поддерево узла в пределах лимита,
        а если и поддерево самого узла больше лимита — только собственный текст узла."""
        if target in self._cards:
            return self.doc_text(target)
        if self._subtree_len[target] <= self.parent_max_chars:
            return self.subtree_text(target)
        return self._nodes[target].text

    # ---------------- опись ----------------

    def inventory(self) -> str:
        if self.use_safe_context:
            from safe_context import clean
            return clean(self._inventory)
        return self._inventory

    def _build_inventory(self) -> str:
        """Опись базы кодом из карточек (03a §3.3): строка на документ; законы — по актам со статьями.
        Если не помещается в лимит — свёртка до групп."""
        cards = list(self._cards.values())
        laws: dict[str, list[str]] = defaultdict(list)
        for c in cards:
            if c.kind == "law":
                laws[c.act or "иные акты"].append(c.number)
        company = sorted((c for c in cards if c.kind != "law"),
                         key=lambda c: (KIND_ORDER.index(c.kind) if c.kind in KIND_ORDER else 99, c.date, c.number))

        def law_lines() -> list[str]:
            lines = []
            for act in sorted(laws):
                nums = sorted({n for n in laws[act] if n}, key=_article_key)
                lines.append(f"- {act}: ст. {', '.join(nums)}" if nums else f"- {act}")
            return lines

        full = [f"В базе знаний {plural_docs(len(cards))}."]
        if company:
            full.append("Документы компании:")
            for c in company:
                full.append(f"- {doc_header(c)}")
        if laws:
            full.append("Законы (только действующие редакции; других статей этих актов в базе нет):")
            full.extend(law_lines())
        text = "\n".join(full)
        if len(text) <= self.inventory_max_chars:
            return text

        # Свёртка: виды документов с количеством и номерами, законы — по актам
        short = [f"В базе знаний {plural_docs(len(cards))}."]
        by_kind: dict[str, list[DocCard]] = defaultdict(list)
        for c in company:
            by_kind[c.kind].append(c)
        for kind in KIND_ORDER:
            if by_kind.get(kind):
                numbers = ", ".join(doc_title(c) for c in by_kind[kind])
                short.append(f"- {KIND_TITLES[kind]} ({len(by_kind[kind])}): {numbers}")
        if laws:
            short.append("Законы (действующие редакции):")
            short.extend(law_lines())
        text = "\n".join(short)
        if len(text) > self.inventory_max_chars:
            text = text[: self.inventory_max_chars - 1].rsplit("\n", 1)[0] + "\n…"
        return text


def embed_nodes(index: Index, client, settings=None, deadline: float | None = None,
                cache: EmbeddingCache | None = None) -> None:
    """Эмбеддинги текстов узлов (search_text) через bge-m3 → index.vectors (float16, L2-норма).

    Кэш по хэшу текста: готовые векторы не пересчитываются. Батчи идут параллельно. Сбой батча или
    дедлайн не роняют сборку: узлы без вектора просто не участвуют в dense-поиске (C2). Попадание
    несовместимой размерности (ключ без ID модели, R6) пересчитывается; не пересчитанное забывается."""
    started = time.time()
    model = getattr(settings, "emb_model", "") or getattr(getattr(client, "s", None), "emb_model", "embedder")
    batch = max(1, getattr(settings, "embed_batch", 64))
    workers = max(1, getattr(settings, "embed_workers", 4))
    if cache is None:
        write = None
        if getattr(settings, "dev_cache", False):
            write = Path(getattr(settings, "dev_cache_dir", ROOT_DIR / "dev_cache")) / "emb_cache.npz"
        cache = EmbeddingCache(model, write_path=write)

    ids = [n.node_id for n in index.searchable_nodes()]
    texts = [index.search_text(nid) for nid in ids]
    shipped = verify_shipped(cache, texts, client, deadline)
    vectors: list[np.ndarray | None] = [cache.get(t) for t in texts]
    todo = [i for i, v in enumerate(vectors) if v is None]
    hits = len(ids) - len(todo)
    failed_batches = 0
    skipped_deadline = 0

    def run(chunk: list[int]) -> tuple[list[int], np.ndarray | None]:
        if deadline is not None and time.time() > deadline:
            return chunk, None
        try:
            embed = getattr(client, "embed_bulk", client.embed)   # пачка индекса: более длинный таймаут (R2)
            return chunk, embed([texts[i] for i in chunk], batch_size=len(chunk))
        except LLMError as exc:
            log(f"[indexer] эмбеддинги батча не получены: {exc}")
            return chunk, None

    chunks = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    retry: list[list[int]] = []
    fresh: set[int] = set()          # индексы, чьи векторы посчитаны эмбеддером в этой сборке

    def accept(chunk: list[int], mat) -> None:
        try:
            rows = iter(mat)
        except TypeError:
            return
        for i, vec in zip(chunk, rows):
            value = stored_vector(vec)
            if value is None:
                continue
            vectors[i] = value
            fresh.add(i)
            cache.put(texts[i], vec)

    if chunks and client is not None:
        with ThreadPoolExecutor(min(workers, len(chunks))) as pool:
            for chunk, mat in pool.map(run, chunks):
                if mat is None:
                    retry.append(chunk)
                else:
                    accept(chunk, mat)
        # Второй проход: упавшие батчи (обычно 429 при параллельной отправке) — по одному, после паузы.
        # Если не прошёл ни один батч — сбой системный (модель недоступна): второй проход только тратит время
        systemic = bool(retry) and len(retry) == len(chunks) and (deadline is None or time.time() < deadline)
        if systemic:
            failed_batches += len(retry)
            log("[indexer] эмбеддер недоступен — индекс без векторов, поиск только полнотекстовый")
            retry = []
        if retry and (deadline is None or time.time() < deadline):
            time.sleep(2.0)
        for chunk in retry:
            chunk_done, mat = run(chunk)
            if mat is not None:
                accept(chunk_done, mat)
            elif deadline is not None and time.time() > deadline:
                skipped_deadline += len(chunk)
            else:
                failed_batches += 1

    # Кэш-попадание несовместимой размерности (ключ без ID модели, R6: смена эмбеддера или чужой кэш)
    # не должно определять dim и вытеснять свежие векторы: такие узлы пересчитываются, а не пересчитанные
    # забываются, чтобы следующая сборка не получила тот же отравленный ключ (кэш — только ускорение, C2).
    dim_mismatch_dropped = 0
    attempted: set[int] = set()
    for _ in range(2):
        fresh_dim = next((vectors[i].shape[0] for i in sorted(fresh) if vectors[i] is not None), 0)
        dim = fresh_dim or next((v.shape[0] for v in vectors if v is not None), 0)
        mismatch = [i for i, v in enumerate(vectors)
                    if v is not None and v.shape[0] != dim and i not in attempted]
        if not mismatch or client is None or (deadline is not None and time.time() > deadline):
            break
        attempted.update(mismatch)
        for start in range(0, len(mismatch), batch):
            chunk, mat = run(mismatch[start:start + batch])
            if mat is not None:
                accept(chunk, mat)
    dim = next((vectors[i].shape[0] for i in sorted(fresh) if vectors[i] is not None), 0) \
        or next((v.shape[0] for v in vectors if v is not None), 0)
    still_bad = [i for i, v in enumerate(vectors) if v is not None and v.shape[0] != dim]
    if still_bad:
        cache.discard([texts[i] for i in still_bad])
        for i in still_bad:
            vectors[i] = None
        dim_mismatch_dropped = len(still_bad)

    if dim:
        mat = np.zeros((len(ids), dim), dtype=np.float16)
        has = np.zeros(len(ids), dtype=bool)
        for i, v in enumerate(vectors):
            if v is not None and v.shape[0] == dim:
                mat[i] = v
                has[i] = True
        index.vec_ids, index.vectors, index.has_vec = ids, mat, has
    if cache.write_path and cache.new:
        cache.save()
    index.emb_stats = {
        "model": model, "nodes": len(ids), "with_vector": int(index.has_vec.sum()) if index.has_vec is not None else 0,
        "cache_hits": hits, "computed": sum(1 for i in fresh if vectors[i] is not None),
        "failed_batches": failed_batches, "skipped_by_deadline": skipped_deadline, "dim": dim,
        "dim_mismatch_dropped": dim_mismatch_dropped,
        "seconds": round(time.time() - started, 2), "cache_sources": cache.loaded_from, "shipped": shipped,
    }


VERIFY_SAMPLE = 5        # сколько готовых векторов пересчитать для проверки совместимости (R6)
VERIFY_MIN_COS = 0.99    # минимальный косинус «готовый вектор — пересчитанный»


def verify_shipped(cache: EmbeddingCache, texts: list[str], client, deadline: float | None) -> dict:
    """Проверка готовых векторов из ZIP (R6): пересчитать VERIFY_SAMPLE узлов (первые по отсортированному ключу —
    выбор воспроизводим) и сравнить косинусы. Все ≥ VERIFY_MIN_COS → векторы приняты. Иначе, а также если
    эмбеддер недоступен или дедлайн индекса прошёл, — все готовые векторы отбрасываются и считаются заново."""
    used = sorted({t for t in texts if cache.is_unverified(t)}, key=text_key)
    invalid = [t for t in used if not usable_vector(cache.get(t))]
    if invalid:
        cache.discard(invalid)
        used = [t for t in used if t not in invalid]
    if not used:
        return {"status": "none", "invalid_vectors_dropped": len(invalid)}
    if client is None or (deadline is not None and time.time() > deadline):
        dropped = cache.drop_unverified()
        log(f"[indexer] готовые векторы не проверить (нет эмбеддера) — отброшены: {dropped}")
        return {"status": "dropped_no_embedder", "dropped": dropped}
    sample = used[:VERIFY_SAMPLE]
    try:
        fresh = client.embed(sample, batch_size=len(sample))
    except LLMError as exc:
        dropped = cache.drop_unverified()
        log(f"[indexer] проверка готовых векторов не удалась ({exc}) — отброшены: {dropped}")
        return {"status": "dropped_no_embedder", "dropped": dropped}
    cosines = []
    try:
        fresh = list(fresh) if fresh is not None else []
    except TypeError:
        fresh = []
    for pos, text in enumerate(sample):
        vec = fresh[pos] if pos < len(fresh) else None
        if not usable_vector(cache.get(text)) or not usable_vector(vec) or stored_vector(vec) is None:
            cosines.append(-1.0)
            continue
        old = cache.get(text).astype(np.float32)
        new = np.asarray(vec, dtype=np.float32)
        if old.shape != new.shape:
            cosines.append(-1.0)
            continue
        denom = float(np.linalg.norm(old) * np.linalg.norm(new)) or 1.0
        cosine = float(old @ new) / denom
        cosines.append(cosine if np.isfinite(cosine) else -1.0)
    min_cos = round(min(cosines), 4)
    if min_cos >= VERIFY_MIN_COS:
        cache.verify(used)
        for text, vec in zip(sample, fresh):
            cache.put(text, vec)
        return {"status": "accepted", "checked": len(sample), "min_cos": min_cos, "vectors": len(used),
                "invalid_vectors_dropped": len(invalid)}
    dropped = cache.drop_unverified()
    for text, vec in zip(sample, fresh):
        cache.put(text, vec)  # пересчитанные векторы образца годны
    log(f"[indexer] готовые векторы несовместимы с эмбеддером (мин. косинус {min_cos}) — отброшены: {dropped}")
    return {"status": "rejected", "checked": len(sample), "min_cos": min_cos, "dropped": dropped,
            "invalid_vectors_dropped": len(invalid)}


def build_index(raw_docs: list[RawDoc], client=None, settings=None, deadline: float | None = None,
                cache: EmbeddingCache | None = None) -> Index:
    """Индекс из любой базы в заданном формате, без LLM-обогащений (срез, 05-tasks этап 1).
    С client — эмбеддинги узлов (T1.8); без него или при сбое эмбеддера — только полнотекстовый поиск (C2)."""
    started = time.time()
    parsed, relations = parse_all(raw_docs, settings)
    index = Index(parsed, relations,
                  parent_max_chars=getattr(settings, "parent_max_chars", 3500),
                  whole_doc_chars=getattr(settings, "whole_doc_chars", 8000),
                  inventory_max_chars=getattr(settings, "inventory_max_chars", 3000),
                  use_safe_context=getattr(settings, "use_safe_context", False))
    if client is not None:
        embed_nodes(index, client, settings, deadline, cache)
    index.build_seconds = round(time.time() - started, 2)
    return index
