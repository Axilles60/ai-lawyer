"""Проверяющий проход (USE_CRITIC): вторая модель читает вопрос, утверждения ответа и тексты положений, на которые они
ссылаются, и решает, нет ли домыслов и не упущены ли условия, исключения или сроки из досье. При замечаниях ответ
перегенерируется с этими замечаниями (pipeline). Дополнительный вызов LLM на вопрос; при сбое шаг пропускается."""

from __future__ import annotations

from dataclasses import dataclass, field

from dialog import history_text
from llm_client import LLMError, Trace

CRITIC_SYSTEM = """Ты — строгий проверяющий ответов юридического ассистента. Тебе дают вопрос, ответ ассистента (утверждения
с цитатами), тексты положений, на которые он ссылается, и перечень других положений досье. Проверь:
1) прямой ответ на вопрос верен и следует из текстов положений;
2) каждое утверждение подтверждено указанным положением: нет домыслов, лишних чисел, выдуманных условий и сроков;
3) не упущено ли положение из перечня, которое меняет, ограничивает или уточняет ответ (условие, исключение, срок,
   сумма, последствия): назови его метку из перечня.
Не придирайся к стилю и длине. Если ответ верен и достаточно полон — «ok».
Верни только JSON: {"verdict": "ok" | "fix", "problems": ["кратко, что не так"], "missing_tags": ["Д1.4"]}."""

NODE_CHARS = 1200      # сколько текста положения показывается проверяющему
LIST_LINES = 60        # сколько других положений досье перечисляется
LIST_CHARS = 9000


@dataclass
class Verdict:
    ok: bool = True
    problems: list[str] = field(default_factory=list)
    missing_tags: list[str] = field(default_factory=list)

    def note(self) -> str:
        parts = ["Проверяющий нашёл замечания к предыдущему ответу."]
        if self.problems:
            parts.append("Замечания: " + "; ".join(self.problems[:5]) + ".")
        if self.missing_tags:
            parts.append("Учти также положения: " + ", ".join(self.missing_tags[:6]) +
                         " (добавь отдельным утверждением со ссылкой и цитатой, если они меняют ответ).")
        parts.append("Исправь ответ: убери утверждения без опоры в тексте положений, не придумывай новых фактов.")
        return " ".join(parts)


def _node_text(index, ctx, tag: str) -> str:
    target = ctx.tag_to_node.get(tag)
    if not target:
        return ""
    try:
        text = index.node(target).text if "." in tag else index.doc_text(target)
        if getattr(index, "use_safe_context", False):
            from safe_context import clean
            text = clean(text)
        return " ".join(text.split())[:NODE_CHARS]
    except Exception:  # неизвестная метка или недоступный источник не создаёт текст доказательства
        return ""


def build_request(index, dialog, router, draft, ctx) -> str:
    history = history_text(dialog)
    parts = []
    if history:
        parts.append(f"ИСТОРИЯ ДИАЛОГА:\n{history}")
    parts.append(f"ВОПРОС:\n{dialog.last_user}")
    lines = []
    cited: list[str] = []
    for i, claim in enumerate(draft.claims, 1):
        lines.append(f"{i}. {claim.text}\n   цитата: «{claim.quote}»")
        for tag in claim.sources:
            if tag not in cited:
                cited.append(tag)
    parts.append("ОТВЕТ АССИСТЕНТА:\n" + "\n".join(lines))
    parts.append("ТЕКСТЫ ПОЛОЖЕНИЙ, НА КОТОРЫЕ ССЫЛАЕТСЯ ОТВЕТ:\n" + "\n".join(
        f"[{tag}] {_node_text(index, ctx, tag)}" for tag in cited))
    others, used = [], 0
    for tag in ctx.tag_to_node:
        if "." not in tag or tag in cited:
            continue
        text = _node_text(index, ctx, tag)[:110]
        if not text:
            continue
        line = f"[{tag}] {text}"
        used += len(line)
        if len(others) >= LIST_LINES or used > LIST_CHARS:
            break
        others.append(line)
    if others:
        parts.append("ПЕРЕЧЕНЬ ДРУГИХ ПОЛОЖЕНИЙ ДОСЬЕ:\n" + "\n".join(others))
    text = "\n\n".join(parts)
    if getattr(index, "use_safe_context", False):
        from safe_context import clean
        text = clean(text)
    return text


def critique(index, dialog, router, draft, ctx, client, trace: Trace | None = None,
             deadline: float | None = None) -> Verdict:
    """Один вызов проверяющего. Неизвестные метки отбрасываются; непонятный ответ считается «ok» (не ломаем ответ)."""
    kwargs = {"deadline": deadline} if deadline is not None else {}
    data = client.chat_json([{"role": "system", "content": CRITIC_SYSTEM},
                             {"role": "user", "content": build_request(index, dialog, router, draft, ctx)}],
                            trace=trace, max_tokens=2048, **kwargs)
    if not isinstance(data, dict):
        return Verdict()
    verdict = str(data.get("verdict") or "ok").strip().lower()
    problems = [str(p).strip() for p in (data.get("problems") or []) if str(p).strip()] \
        if isinstance(data.get("problems"), list) else []
    tags = [str(t).strip().strip("[]") for t in (data.get("missing_tags") or [])] \
        if isinstance(data.get("missing_tags"), list) else []
    known = [t for t in dict.fromkeys(tags) if t in ctx.tag_to_node and "." in t]
    if verdict != "fix" or not (problems or known):
        return Verdict()
    return Verdict(ok=False, problems=problems, missing_tags=known)


__all__ = ["CRITIC_SYSTEM", "Verdict", "critique", "build_request", "LLMError"]
