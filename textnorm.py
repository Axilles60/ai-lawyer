"""Единая нормализация текста для индекса, запроса и сверки цитат (T1.5; 03a §3.2).

Один и тот же код формирует токены для полнотекстового индекса и для запроса — иначе спецтокены
номеров не совпадут (06-implementation-guide §10).

- tokens(text): нижний регистр, ё→е, числа с пробелами-разделителями тысяч склеиваются («450 000» → 450000),
  стоп-слова убираются, лёгкое усечение окончаний русских слов (стеммера в разрешённых библиотеках нет, RQ-2).
- special_tokens(text): буквенно-цифровые токены реквизитов — их понимает токенизатор FTS5 unicode61:
    номер документа «№ П-17/2024» → noп172024;  пункт «п. 3.2» / «3.2.» в начале узла → p3x2;
    подпункт «пп. 1» → pp1;  статья «ст. 70» / «статья 13.11» → st70 / st13x11;  часть «ч. 2» / «часть вторая» → ch2;
    ИНН → inn<цифры>.
- norm_quote(text): форма для сверки дословных цитат.
"""

from __future__ import annotations

import re

STOPWORDS = {
    "и", "в", "во", "на", "по", "с", "со", "к", "ко", "о", "об", "обо", "от", "до", "из", "за", "у", "для",
    "при", "про", "под", "над", "без", "а", "но", "или", "ли", "же", "бы", "не", "ни", "что", "чтобы", "как",
    "это", "этот", "эта", "эти", "то", "тот", "та", "те", "так", "также", "его", "ее", "их", "он", "она",
    "они", "оно", "мы", "вы", "я", "ты", "нас", "вас", "нам", "вам", "мне", "все", "весь", "всё", "был",
    "была", "было", "были", "быть", "есть", "уже", "еще", "ещё", "если", "когда", "где", "кто", "который",
    "которая", "которое", "которые", "какой", "какая", "какие", "можно", "нужно", "ли",
}

# Окончания по убыванию длины; отрезаем самое длинное, если основа остаётся не короче MIN_STEM.
ENDINGS = sorted({
    "иями", "ями", "ами", "иях", "ях", "ах", "ией", "ием", "ией", "ого", "его", "ому", "ему", "ыми", "ими",
    "ыми", "ой", "ей", "ий", "ый", "ая", "яя", "ое", "ее", "ые", "ие", "ую", "юю", "ом", "ем", "ам", "ям",
    "ов", "ев", "ия", "ья", "ье", "ии", "ью", "ию", "ым", "им", "ешь", "ете", "ить", "ать", "ять", "еть", "уть", "ут",
    "ют", "ит", "ат", "ят", "ет", "ла", "ли", "ло", "ть", "ых", "их", "ой", "ую", "и", "ы", "а", "я", "о",
    "е", "у", "ю", "ь", "й", "ость", "ости", "остью", "ание", "ания", "анию", "анием", "ение", "ения",
    "ению", "ением", "енный", "енного", "енной", "ованн", "ующ", "ющ", "ащ", "ящ",
}, key=len, reverse=True)
MIN_STEM = 4

ORDINALS = {
    "перв": 1, "втор": 2, "трет": 3, "четверт": 4, "пят": 5, "шест": 6, "седьм": 7, "восьм": 8,
    "девят": 9, "десят": 10, "одиннадцат": 11, "двенадцат": 12, "тринадцат": 13, "четырнадцат": 14,
    "пятнадцат": 15, "шестнадцат": 16, "семнадцат": 17, "восемнадцат": 18, "девятнадцат": 19, "двадцат": 20,
}
_ORD_RE = "|".join(sorted(ORDINALS, key=len, reverse=True))

WORD_RE = re.compile(r"[a-zа-я0-9]+")
THOUSANDS_RE = re.compile(r"(?<![\d.,])(\d{1,3})((?:[  ]\d{3})+)(?![\d])")
NUM_RE = r"\d+(?:\.\d+)*(?:-\d+)?"

DOC_NUMBER_RE = re.compile(r"(?:№|\bN)\s*([A-Za-zА-Яа-яЁё0-9](?:[A-Za-zА-Яа-яЁё0-9\-/.]*[A-Za-zА-Яа-яЁё0-9])?)")
# Номер документа без «№»: буквы-цифры с разделителем и годом — «П-17/2024», «ПТ-05/2023», «А-08-2024»
BARE_DOC_NUMBER_RE = re.compile(r"(?<![\w/])([A-Za-zА-Яа-яЁё]{1,4}-?\d{1,5}[/\-]\d{2,4})(?![\w/])")
POINT_RE = re.compile(rf"(?<![а-яa-z])(?:п\.|пункт\w*)\s*({NUM_RE})", re.IGNORECASE)
SUBPOINT_RE = re.compile(rf"(?<![а-яa-z])(?:пп\.|подпункт\w*)\s*({NUM_RE})", re.IGNORECASE)
ARTICLE_RE = re.compile(rf"(?<![а-яa-z])(?:ст\.|стать\w*)\s*({NUM_RE})", re.IGNORECASE)
PART_RE = re.compile(rf"(?<![а-яa-z])(?:ч\.|част\w*)\s*({NUM_RE})", re.IGNORECASE)
PART_ORD_RE = re.compile(rf"(?<![а-яa-z])част\w*\s+({_ORD_RE})\w*|(?<![а-яa-z])({_ORD_RE})\w*\s+част", re.IGNORECASE)
LEADING_POINT_RE = re.compile(r"^\s*(\d+(?:\.\d+)+)\.?\s")
INN_RE = re.compile(r"ИНН\s*:?\s*(\d{10,12})", re.IGNORECASE)


def _lower(text: str) -> str:
    return (text or "").lower().replace("ё", "е")


def stem(word: str) -> str:
    if len(word) <= MIN_STEM or not re.search(r"[а-я]", word):
        return word
    if word.endswith(("ся", "сь")) and len(word) - 2 >= MIN_STEM:
        word = word[:-2]
    for ending in ENDINGS:
        if word.endswith(ending) and len(word) - len(ending) >= MIN_STEM:
            return word[: -len(ending)]
    return word


def tokens(text: str) -> list[str]:
    """Нормализованные основы слов и числа (без спецтокенов)."""
    text = THOUSANDS_RE.sub(lambda m: m.group(1) + re.sub(r"\D", "", m.group(2)), text or "")
    out = []
    for w in WORD_RE.findall(_lower(text)):
        if w in STOPWORDS:
            continue
        out.append(stem(w))
    return out


def _num_token(prefix: str, num: str) -> str:
    return prefix + num.replace(".", "x").replace("-", "y")


def doc_number_token(number: str) -> str:
    return "no" + re.sub(r"[^a-zа-я0-9]", "", _lower(number))


def special_tokens(text: str) -> list[str]:
    """Буквенно-цифровые токены реквизитов (без повторов, в порядке появления)."""
    text = text or ""
    found: list[str] = []
    for m in DOC_NUMBER_RE.finditer(text):
        found.append(doc_number_token(m.group(1)))
    for m in BARE_DOC_NUMBER_RE.finditer(text):
        found.append(doc_number_token(m.group(1)))
    for m in SUBPOINT_RE.finditer(text):
        found.append(_num_token("pp", m.group(1)))
    for m in POINT_RE.finditer(text):
        found.append(_num_token("p", m.group(1)))
    m = LEADING_POINT_RE.match(text)
    if m:
        found.append(_num_token("p", m.group(1)))
    for m in ARTICLE_RE.finditer(text):
        found.append(_num_token("st", m.group(1)))
    for m in PART_RE.finditer(text):
        found.append(_num_token("ch", m.group(1)))
    for m in PART_ORD_RE.finditer(text):
        word = _lower(m.group(1) or m.group(2))
        for stem_word, value in ORDINALS.items():
            if word.startswith(stem_word):
                found.append(f"ch{value}")
                break
    for m in INN_RE.finditer(text):
        found.append("inn" + m.group(1))
    seen, out = set(), []
    for t in found:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def index_tokens(text: str) -> list[str]:
    """Всё, что идёт в полнотекстовый индекс и в запрос: основы + спецтокены."""
    return tokens(text) + special_tokens(text)


QUOTE_CHARS = str.maketrans({"«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "‘": "'", "’": "'",
                             "‐": "-", "‑": "-", "–": "-", "—": "-", "­": None, " ": " "})


def norm_quote(text: str) -> str:
    """Форма для сверки дословных цитат: регистр, ё, кавычки, тире, пробелы."""
    text = _lower(text).translate(QUOTE_CHARS)
    return re.sub(r"\s+", " ", text).strip(" .;,")
