"""Защитные проверки (03b §1.1, §5; P5.2, P5.9).

T1.9 — N1, regex-guard: дешёвый первый слой против манипуляций в вопросе. Он только ставит флаги —
подсказку роутеру (T1.10) и пометку в трейсе; сам ничего не блокирует: реплика с давлением обычно
содержит и настоящий юридический вопрос, на который нужно ответить по базе (Q15).

T1.15 — проверки ответа V1/V2/V10, сборка текста и ссылок кодом, запасной ответ (ниже).
"""

from __future__ import annotations

import re

GUARD_PATTERNS: dict[str, re.Pattern] = {
    # «Игнорируй предыдущие инструкции», «забудь правила», «ignore previous instructions»
    "instruction_override": re.compile(
        r"(игнорируй\w*|проигнорируй\w*|забудь\w*|не\s+учитывай\w*|отмени(?:те)?|отключи(?:те)?|"
        r"обойди(?:те)?|нарушь(?:те)?)\s+"
        r"(?:\w+\s+){0,3}(инструкц\w*|правил\w*|указани\w*|ограничени\w*|промпт\w*|настройк\w*)"
        r"|ignore\s+(?:all\s+|the\s+|any\s+)?(?:previous|prior|above|your)?\s*(instruction|rule|prompt)s?"
        r"|disregard\s+(?:all\s+|the\s+|previous\s+)*(instruction|rule|prompt)s?",
        re.IGNORECASE),
    # Смена роли и «режимы без ограничений»
    "role_change": re.compile(
        r"(ты\s+теперь|теперь\s+ты|отныне\s+ты|представь,?\s+что\s+ты|притворись|веди\s+себя\s+как|"
        r"выйди\s+из\s+роли|сыграй\s+роль|режим\s+(разработчика|без\s+ограничений|бога)|"
        r"you\s+are\s+now|act\s+as|pretend\s+(to\s+be|you)|developer\s+mode|jailbreak|\bDAN\b)",
        re.IGNORECASE),
    # Поддельные системные сообщения внутри вопроса
    "fake_system": re.compile(
        r"(\[\s*(system|система|sys)\b|<\s*/?\s*system\s*>|^\s*(система|system|admin|администратор)\s*:|"
        r"системн\w+\s+(сообщени|промпт|инструкци|указани)\w*|#{2,}\s*system|новые\s+инструкци\w*|"
        r"обновлени\w+\s+(правил|инструкци)\w*|сообщение\s+от\s+(разработчик|администратор)\w*)",
        re.IGNORECASE | re.MULTILINE),
    # Просьба раскрыть промпт и правила
    "prompt_leak": re.compile(
        r"((покажи|выведи|раскрой|напечатай|перескажи)\s+(\w+\s+){0,2}(промпт\w*|инструкци\w*|"
        r"системн\w+\s+сообщени\w*)|reveal\s+(your\s+)?(system\s+)?prompt|print\s+(your\s+)?instructions)",
        re.IGNORECASE),
    # Давление: изменить подтверждённый ответ, ответить заданным текстом
    "pressure": re.compile(
        r"(исправь\s+(свой\s+|твой\s+)?(ответ|на\s)|ты\s+(ошибся|ошибаешься|неправ)|"
        r"(ответь|напиши|скажи|подтверди),?\s+(только\s+)?(что|«|\")|"
        r"просто\s+(согласись|подтверди)|не\s+спорь|я\s+лучше\s+знаю|"
        r"(начальник|руководитель|директор)\w*\s+(велел|сказал|приказал)\w*\s+(ответить|написать))",
        re.IGNORECASE),
    # Закодированные вставки (обфускация, P5.1)
    "encoded": re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{48,}={0,2}(?![A-Za-z0-9+/])"),
    # Раскрытие промпта через указание на расположение («текст выше», «весь промпт»)
    "prompt_exfil_location": re.compile(
        r"(текст|сообщени\w*|промпт\w*|инструкци\w*|правил\w*)\s+(выше|над\s+этим|перед\s+этим|"
        r"в\s+начале\s+диалога)|(весь|всю|все)\s+(промпт|инструкци\w*|системн\w+\s+сообщени\w*|"
        r"исходн\w*\s+(код|текст))|(повтори\w*|выведи\w*|покажи\w*|процитируй\w*|назови\w*|"
        r"распечатай\w*|напечатай\w*|прочитай\w*|скажи\w*)\s+"
        r"(весь\s+|сво\w+\s+|тво\w+\s+)?(системн\w*\s+)?(промпт\w*|инструкци\w*)\s+"
        r"(дословно|целиком|полностью|выше)",
        re.IGNORECASE),
    # Социальная утечка: сведения «из твоих инструкций/промпта» (оплата, секреты, ключи)
    "prompt_leak_social": re.compile(
        r"(заплатили|заплатят|получил\w*|заработал\w*|спрятан\w*|скрыт\w*).{0,60}?"
        r"(из|в)\s+(твоих|своих)\s+(инструкци\w*|промпт\w*|системн\w*\s+сообщени\w*)"
        r"|(из|в)\s+(твоих|своих)\s+(инструкци\w*|промпт\w*).{0,40}?"
        r"(сумм\w*|секрет\w*|ключ\w*|парол\w*|токен\w*|оплат\w*)",
        re.IGNORECASE),
    # «Отвечай как ИИ/ассистент без ограничений» (без слова «режим»)
    "no_limits_persona": re.compile(
        r"(как\s+)?(ИИ|ии|модель|ассистент\w*|помощник\w*|чат-?бот\w*)\s+без\s+(ограничени\w*|цензур\w*|правил)"
        r"|(отвечай|веди\s+себя|работай|действуй)\s+без\s+(ограничени\w*|цензур\w*|правил)",
        re.IGNORECASE),
    # Правка собственных правил/инструкций («добавь в свои правила», «измени свои инструкции»)
    "rules_edit": re.compile(
        r"(добавь\w*|внеси\w*|запиши\w*|измени\w*|обнови\w*|замени\w*|удали\w*|отмени\w*)\s+"
        r"(в\s+)?(сво\w+|тво\w+)\s+(правил\w*|инструкци\w*|ограничени\w*|настройк\w*)"
        r"|(сво\w+|тво\w+)\s+(правил\w*|инструкци\w*)\s+(теперь\s+)?(таковы|следующи\w*|гласят)",
        re.IGNORECASE),
    # Поддельная реплика ассистента внутри текста пользователя (после нормализации
    # метки реплик удаляются — остаток «Ассистент:» в тексте реплики подозрителен)
    "fake_assistant_label": re.compile(
        r"^\s*(ассистент|assistant|ai|ии)\s*[:：]\s*\S", re.IGNORECASE | re.MULTILINE),
    # Эскалация привилегий: выполнить команду/действие от имени администратора системы
    "privilege_exec": re.compile(
        r"(выполни\w*|запусти\w*|введи\w*|отправь\w*)\s+(\w+\s+){0,3}(sudo\b|\w+\s+от\s+имени\s+"
        r"(администратор\w*|root)|как\s+(администратор|root)\s+(системы|хоста|сервера))",
        re.IGNORECASE),
}


def guard_flags(text: str) -> list[str]:
    """Флаги манипуляций в тексте (по порядку объявления); пустой список — ничего не найдено."""
    return [name for name, pattern in GUARD_PATTERNS.items() if pattern.search(text or "")]


def guard_matches(text: str) -> dict[str, str]:
    """Сработавшие фрагменты — для трейса (без изменения поведения)."""
    out = {}
    for name, pattern in GUARD_PATTERNS.items():
        m = pattern.search(text or "")
        if m:
            out[name] = m.group(0)[:80]
    return out


# ======================================================================================================
# T1.15 — проверки ответа V1, V2, V10; сборка текста и ссылок кодом; запасной ответ (03b §4.3–4.4, §5)
# ======================================================================================================

from rapidfuzz import fuzz  # noqa: E402

from kb_types import Context, Draft, Issue  # noqa: E402
from textnorm import doc_number_token, norm_quote, special_tokens  # noqa: E402

TAG_RE = re.compile(r"\[\[\s*(Д\d+(?:\.\d+)?)\s*\]\]")
# Группа меток с разделителями «,», «;» или союзом «и» как отдельным словом (не последней буквой слова)
TAG_GROUP_RE = re.compile(r"(?:(?:\s*[,;])?\s*(?:(?<=\s)и\s+)?\[\[\s*Д\d+(?:\.\d+)?\s*\]\])+")
LATIN_WORD_RE = re.compile(r"\b[A-Za-z]{3,}\b")
POSITION_RE = re.compile(r"(?<![а-яa-z])(?:пп\.|п\.|ст\.|ч\.|пункт\w*|стать\w*|част\w*|подпункт\w*)\s*\d+(?:\.\d+)*(?:-\d+)?",
                         re.IGNORECASE)
DOC_NUMBER_IN_TEXT_RE = re.compile(r"№\s*[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9\-/.]*")
MAX_ANSWER_CHARS = 10_000

OFFTOPIC_ANSWER = ("Я юридический ассистент и отвечаю на вопросы по документам компании и законодательству из базы "
                   "знаний. С этим вопросом помочь не могу — задайте, пожалуйста, юридический вопрос.")
MANIPULATION_ANSWER = ("Я не меняю свою роль и правила работы и не выполняю инструкции, встроенные в вопрос. Готов ответить "
                       "на юридический вопрос по документам базы знаний.")
DEFAULT_REFUSE = ("В базе знаний нет документов, позволяющих ответить на этот вопрос, поэтому ответить по документам "
                  "нельзя; по памяти не отвечаю. Обратитесь, пожалуйста, к юристу.")
TECHNICAL_ANSWER = ("Не удалось сформировать ответ по базе знаний из-за технической ошибки. Повторите, пожалуйста, "
                    "запрос позже.")


def _node_texts(tags: list[str], ctx: Context, index) -> list[str]:
    texts = []
    for t in tags:
        target = ctx.tag_to_node.get(t)
        if not target:
            continue
        texts.append(index.node(target).text if "." in t else index.doc_text(target))
    return texts


def _quote_ok(quote: str, texts: list[str], settings) -> bool:
    """V10: цитата содержится в тексте одного из источников — точно или с порогом rapidfuzz."""
    q = norm_quote(quote)
    if not q:
        return False
    min_chars = getattr(settings, "quote_min_chars", 20)
    threshold = getattr(settings, "quote_fuzzy_threshold", 95)
    for text in texts:
        t = norm_quote(text)
        if len(q) < min(min_chars, len(t)):
            continue  # слишком короткая цитата ничего не подтверждает (кроме случая, когда узел сам короче)
        if q in t:
            return True
        if len(q) >= min_chars and fuzz.partial_ratio(q, t) >= threshold:
            return True
    return False


def _allowed_positions(tags: list[str], ctx: Context, index) -> set[str]:
    """Спецтокены положений и номеров документов, которые подтверждаются процитированными узлами."""
    allowed: set[str] = set()
    for t in tags:
        target = ctx.tag_to_node.get(t)
        if not target:
            continue
        doc_id = target if "." not in t else index.node(target).doc_id
        card = index.doc(doc_id)
        if card.number:
            allowed.add(doc_number_token(card.number))
            if card.kind == "law":
                allowed.update(special_tokens(f"ст. {card.number}"))
        if "." in t:
            node = index.node(target)
            allowed.update(special_tokens(node.label))
            allowed.update(special_tokens(node.text))      # ссылки внутри положения («части 2 статьи 10»)
            allowed.update(special_tokens(node.lead_in))
        else:
            allowed.update(special_tokens(index.doc_text(doc_id)))
    return allowed


def check(draft: Draft, ctx: Context, index, settings=None) -> list[Issue]:
    """V1 — структура и язык; V2 — метки и номера положений/документов в тексте; V10 — дословные цитаты."""
    issues: list[Issue] = []
    if getattr(settings, "use_answer_variants", False) and draft.decision == "clarify":
        issues.append(Issue("V1", None, "Q19: нужен условный содержательный answer либо обоснованный refuse, не clarify"))
    if draft.decision not in ("answer", "refuse", "clarify"):
        issues.append(Issue("V1", None, f"недопустимое решение {draft.decision!r}"))
    if draft.decision == "answer" and not draft.claims:
        issues.append(Issue("V1", None, "решение «answer» без утверждений"))
    for i, claim in enumerate(draft.claims):
        if not claim.sources:
            issues.append(Issue("V1", i, "у утверждения нет меток источников [[Д…]]"))
        latin = LATIN_WORD_RE.findall(claim.text)
        if latin:
            issues.append(Issue("V1", i, f"слова не на русском: {', '.join(latin[:3])}"))
        if len(claim.text) > 1500:
            issues.append(Issue("V1", i, "утверждение слишком длинное"))
        unknown = [t for t in claim.sources + TAG_RE.findall(claim.text) if t not in ctx.tag_to_node]
        if unknown:
            issues.append(Issue("V2", i, f"метки не из досье: {', '.join(sorted(set(unknown)))}"))
        plain = TAG_RE.sub(" ", claim.text)
        written: set[str] = set()
        for m in POSITION_RE.finditer(plain):
            written.update(t for t in special_tokens(m.group(0)) if not t.startswith("no"))
        for m in DOC_NUMBER_IN_TEXT_RE.finditer(plain):
            written.update(t for t in special_tokens(m.group(0)) if t.startswith("no"))
        if written:
            stray = sorted(written - _allowed_positions(claim.sources, ctx, index))
            if stray:
                issues.append(Issue("V2", i, f"положения/номера не подтверждены источниками: {', '.join(stray)}"))
        if claim.sources and not unknown:
            texts = _node_texts(claim.sources, ctx, index)
            if not _quote_ok(claim.quote, texts, settings):
                issues.append(Issue("V10", i, "цитата не найдена дословно в тексте указанных положений"))
            elif getattr(settings, "use_safe_context", False):
                from safe_context import clean
                # Если очистка изменила источник, цитата должна точно присутствовать и в видимом тексте.
                q = norm_quote(claim.quote)
                if not any(_quote_ok(claim.quote, [text], settings) if clean(text) == text
                           else bool(q and q in norm_quote(clean(text))) for text in texts):
                    issues.append(Issue("V10", i, "цитата не найдена в показанной части источника"))
    return issues


def issues_note(issues: list[Issue], draft: Draft) -> str:
    """Замечания для повторной генерации — конкретные, по утверждениям."""
    lines = []
    for issue in issues:
        where = f"утверждение {issue.claim_index + 1}" if issue.claim_index is not None else "ответ"
        if issue.claim_index is not None and issue.claim_index < len(draft.claims):
            where += f" («{draft.claims[issue.claim_index].text[:80]}»)"
        lines.append(f"- {issue.check}, {where}: {issue.message}")
    lines.append("Исправь: только метки из досье, номера положений и документов сам не пиши, quote — точная копия "
                 "текста узла, текст только на русском.")
    return "\n".join(lines)


# ---------------- сборка текста и ссылок кодом (03b §4.3) ----------------

def _reference_text(text: str, index) -> str:
    """Clean only generated source metadata; preserve source documents and the OFF path."""
    if getattr(index, "use_safe_context", False):
        from safe_context import clean
        return clean(text)
    return text


def _doc_reference(card, index) -> str:
    """«договор аренды нежилого помещения № А-08/2024 от 01.08.2024 с ООО «Северный Склад»»; закон — «ТК РФ»."""
    from indexer import doc_title  # локальный импорт: модуль используется и без индекса (guard_flags)
    if card.kind == "law":
        return _reference_text(card.act or doc_title(card), index)
    title = doc_title(card)
    title = title[0].lower() + title[1:] if title else title
    own = index.own_org()
    others = [name for name, role, _ in card.parties if role and role != "издатель"
              and not (own and name == own.canonical)]
    return _reference_text(f"{title} с {others[0]}" if others else title, index)


def _position(tag: str, ctx: Context, index, with_doc: bool) -> str:
    target = ctx.tag_to_node[tag]
    if "." not in tag:
        return _doc_reference(index.doc(target), index)
    node = index.node(target)
    card = index.doc(node.doc_id)
    if card.kind == "law":
        return _reference_text(f"{card.act}, {node.label}" if node.label else card.act, index)
    if with_doc:
        ref = _doc_reference(card, index)
        return _reference_text(f"{ref}, {node.label}" if node.label else ref, index)
    return _reference_text(node.label, index)  # без номера — документ назван в «Источник»


_SAFE_LOCAL_NOTICE = "Изолированный фрагмент не показан в доступном контексте; подтвердить его текст автоматически не удалось."


def _safe_missing(missing: str, ctx: Context, index) -> str:
    """Не превращать изолированный фрагмент в утверждение об отсутствии во всей базе."""
    if not getattr(index, "use_safe_context", False):
        return missing
    isolated = any("[изолированная инструкция]" in block[3] for block in ctx.blocks)
    if not isolated:
        return missing
    if not missing:
        return _SAFE_LOCAL_NOTICE
    sentences = re.split(r"(?<=[.!?])\s+", missing.strip())
    result = []
    for sentence in sentences:
        mentions_fragment = re.search(r"(?i)(?:изолир|цитат|фрагмент|не\s+полностью|не\s+показан)", sentence)
        global_absence = re.search(r"(?i)в\s+базе[^.!?]{0,120}(?:нет|отсутств|не\s+привед|не\s+содерж)", sentence)
        result.append(_SAFE_LOCAL_NOTICE if mentions_fragment and global_absence else sentence)
    return " ".join(result)


def render(draft: Draft, ctx: Context, index) -> str:
    """Итоговый текст: утверждения по порядку с подставленными ссылками + оговорка из missing (FR-13).
    Один документ компании — «(п. 3.1)» в тексте и «Источник: …» в конце; несколько — документ в каждой ссылке;
    закон — «(ТК РФ, ст. 70, ч. 5)»."""
    if draft.decision == "clarify":
        text = f"Уточните, пожалуйста: {draft.missing}" if draft.missing else \
            "Уточните, пожалуйста, о каком документе или периоде идёт речь."
        return finalize(text)
    if draft.decision == "refuse" or not draft.claims:
        return finalize(_safe_missing(draft.missing, ctx, index) or DEFAULT_REFUSE)

    used = [t for c in draft.claims for t in c.sources if t in ctx.tag_to_node]
    docs: list[str] = []
    for t in used:
        target = ctx.tag_to_node[t]
        d = target if "." not in t else index.node(target).doc_id
        if d not in docs:
            docs.append(d)
    company_docs = [d for d in docs if index.doc(d).kind != "law"]
    single_company = len(company_docs) == 1
    sentences = []
    for claim in draft.claims:
        def replace(m: re.Match) -> str:
            refs = []
            for t in TAG_RE.findall(m.group(0)):
                if t in ctx.tag_to_node:
                    ref = _position(t, ctx, index, with_doc=not single_company)
                    if ref and ref not in refs:
                        refs.append(ref)
            return f" ({'; '.join(refs)})" if refs else ""
        text = TAG_GROUP_RE.sub(replace, claim.text)
        text = re.sub(r"\s+([.,;:!?])", r"\1", text).strip()
        if text and text[-1] not in ".!?":
            text += "."
        sentences.append(text)
    if single_company:
        card = index.doc(company_docs[0])
        labels = []
        for t in used:
            if "." in t and index.node(ctx.tag_to_node[t]).doc_id == card.doc_id:
                label = _reference_text(index.node(ctx.tag_to_node[t]).label, index)
                if label and label not in labels:
                    labels.append(label)
        source = _doc_reference(card, index)
        source = source[0].upper() + source[1:]
        sentences.append(f"Источник: {source}" + (f", {', '.join(labels)}." if labels else "."))
    if draft.missing:
        missing = _safe_missing(draft.missing.strip(), ctx, index)
        sentences.append(missing if missing.endswith((".", "!", "?")) else missing + ".")
    return finalize(" ".join(sentences))


def finalize(text: str) -> str:
    """Одна физическая строка, без служебных меток, не длиннее лимита платформы (REG-6)."""
    text = TAG_RE.sub("", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_ANSWER_CHARS]


def prune(draft: Draft, issues: list[Issue]) -> Draft:
    """Удаление утверждений, не прошедших V1/V2/V10 (03b §4.3) — без повторного вызова модели."""
    bad = {i.claim_index for i in issues if i.claim_index is not None}
    return Draft(decision=draft.decision, claims=[c for i, c in enumerate(draft.claims) if i not in bad],
                 missing=draft.missing)


def extractive_fallback(ctx: Context | None, index, best_node_id: str | None = None) -> str:
    """Запасной ответ (03b §4.4): лучший найденный узел дословно со ссылкой — не заглушка и не выдумка."""
    node_id = best_node_id or (ctx.best_node_id if ctx else None)
    if not node_id:
        return TECHNICAL_ANSWER
    try:
        node = index.node(node_id)
    except KeyError:
        return TECHNICAL_ANSWER
    text = node.text.strip()
    if getattr(index, "use_safe_context", False):
        from safe_context import clean
        text = clean(text)
    if len(text) > 700:
        text = text[:700].rsplit(" ", 1)[0] + "…"
    ref = _doc_reference(index.doc(node.doc_id), index)
    if node.label:
        ref = f"{ref}, {node.label}"
    return finalize(f"Наиболее близкое к вопросу положение базы: «{text}» ({ref}). Сформировать ответ автоматически "
                    f"не удалось — проверьте применимость положения.")
