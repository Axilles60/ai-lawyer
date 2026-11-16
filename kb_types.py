"""Общие структуры данных решения (06-implementation-guide §6).

Поля — минимально необходимые; добавлять можно, переименовывать после появления
зависящего кода — нет.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------- загрузка и индекс (оффлайн) ----------------

@dataclass(frozen=True)
class RawDoc:
    """Документ базы знаний после чтения из архива (T1.1)."""
    name: str      # путь внутри архива (декодированный)
    suffix: str    # .txt | .md | .docx
    text: str      # нормализованный текст (переводы строк \n, без BOM и неразрывных пробелов)
    sha256: str    # хэш нормализованного текста — для сверки с предрасчитанным индексом (Q4a)


@dataclass
class DocCard:
    doc_id: str
    file: str
    sha256: str
    kind: str = "other"           # contract | amendment | regulation | letter | memo | law | other
    act: str = ""                 # вид акта / документа
    number: str = ""
    date: str = ""                # ISO yyyy-mm-dd, если удалось распознать
    title: str = ""
    parties: list[tuple[str, str, str]] = field(default_factory=list)  # (имя, роль, ИНН)
    status: str = ""              # действует | утратил силу | ...
    valid_from: str = ""
    valid_to: str = ""
    condition: str = ""           # условие применения, текстом
    source_url: str = ""
    export_date: str = ""
    chars: int = 0
    whole: bool = False           # документ отдаётся в контекст целиком (03a §2.2)
    act_full: str = ""            # для законов: полное название акта («Трудовой кодекс Российской Федерации»)


@dataclass
class Node:
    node_id: str
    doc_id: str
    parent_id: str | None
    level: int
    path: str                     # «3.2», «ст.70/ч.2», «ст.164/п.2/пп.1»
    label: str                    # готовый кусок ссылки: «п. 3.2», «ст. 70, ч. 2»
    heading: str = ""
    text: str = ""
    lead_in: str = ""             # вводная фраза родителя (03a §2.1, U-7)
    start: int = 0
    end: int = 0
    flags: list[str] = field(default_factory=list)  # editorial_note, instruction_like, repealed, split, fragment
    notes: list[str] = field(default_factory=list)  # редакционные пометки, относящиеся к узлу (Q9) — не текст нормы

    provenance: dict[str, Any] = field(default_factory=dict)  # 03a §2.1.1, separate from source/embedding text
    contextual_text: str = ""      # LLM-generated search hint; never primary evidence or citation text


@dataclass
class ParsedDoc:
    card: DocCard
    nodes: list[Node]
    link_hints: list[str] = field(default_factory=list)  # «к договору № …», «утратил силу с …»
    text: str = ""                # полный нормализованный текст документа (из RawDoc)
    warnings: list[str] = field(default_factory=list)  # предупреждения разбора: нумерация, потери текста


@dataclass(frozen=True)
class Relation:
    src_doc: str
    dst_doc: str
    kind: str                     # amends | replaces | supplements
    date: str = ""
    condition: str = ""


@dataclass
class Entity:
    entity_id: str
    kind: str                     # org | document | act
    canonical: str
    inn: str = ""
    aliases: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    roles: dict[str, str] = field(default_factory=dict)  # doc_id -> роль стороны («Арендатор», «издатель»)
    own: bool = False             # «наша» организация: сторона большинства документов компании («мы», «наш»)


@dataclass(frozen=True)
class EntityMatch:
    entity_id: str
    kind: str
    canonical: str
    doc_ids: tuple[str, ...]
    score: int                    # 100 — полное совпадение алиаса; меньше — частичное
    matched: str                  # какой алиас или слово сработали
    full: bool


# ---------------- онлайн ----------------

@dataclass(frozen=True)
class Turn:
    role: str                     # user | assistant
    text: str


@dataclass
class Dialog:
    """Вход вопроса после разбора (N0): история и последняя реплика пользователя."""
    raw: str
    turns: list[Turn] = field(default_factory=list)
    last_user: str = ""           # реплика, на которую отвечаем
    history: list[Turn] = field(default_factory=list)  # всё до последней реплики пользователя
    guard: list[str] = field(default_factory=list)      # флаги regex-guard (N1): подсказка роутеру, не блок

    @property
    def is_dialog(self) -> bool:
        return bool(self.history)

@dataclass
class RouterResult:
    scope: str = "legal"          # legal | offtopic
    manipulation: bool = False
    illegal: bool = False
    query: str = ""               # самостоятельный очищенный запрос
    doc_refs: list[str] = field(default_factory=list)
    node_refs: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    event_date: str = ""
    user_claims: list[str] = field(default_factory=list)
    subqueries: list[str] = field(default_factory=list)  # кандидат S: аспекты вопроса для отдельного поиска
    stepback: str = ""        # кандидат T3.30: более общий вопрос «шаг назад» для retrieval


@dataclass
class Context:
    blocks: list[tuple[str, str, bool, str]] = field(default_factory=list)  # (тег «Д1», doc_id, whole, текст)
    tag_to_node: dict[str, str] = field(default_factory=dict)
    inventory: str = ""
    notes: list[str] = field(default_factory=list)
    best_node_id: str | None = None
    ambiguous: list[str] = field(default_factory=list)


@dataclass
class Claim:
    text: str                     # содержит метки [[Д1.3.1]]
    sources: list[str] = field(default_factory=list)
    quote: str = ""


@dataclass
class Draft:
    decision: str = "answer"      # answer | refuse | clarify
    claims: list[Claim] = field(default_factory=list)
    missing: str = ""


@dataclass(frozen=True)
class Issue:
    check: str                    # V1 | V2 | V10
    claim_index: int | None
    message: str


@dataclass
class Result:
    answer: str
    decision: str = "answer"
    claims: list[Claim] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)
    fallback: bool = False
    issues: list[Issue] = field(default_factory=list)
    trace: Any = None
    mode: str = "full"            # full | light | extractive | technical (R4, 03b §4.4)
    missing: str = ""             # поле missing ответа модели: подсказка для повторного цикла RAG (режим LLM-only)
