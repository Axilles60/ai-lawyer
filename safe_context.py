"""SAFE_CONTEXT: ограниченная изоляция явно адресованных модели инструкций.

Диапазоны задаются в координатах входной строки; оригинал не изменяется.
Детектор не предназначен для распознавания всех способов инъекции.
"""
from dataclasses import dataclass, replace
import re

from kb_types import Turn

MARKER = "\n[изолированная инструкция]\n"
# Нормируем только известные управляющие глаголы с разделёнными буквами; диапазоны остаются исходными.
_SPACED_VERBS = ('игнорируй', 'забудь', 'отмени', 'ignore', 'forget', 'disregard')
_SPACED_PATTERN = '|'.join(r'[\s.\u200b_-]+'.join(re.escape(c) for c in verb) for verb in _SPACED_VERBS)
_RULES = (
    ('spaced_command', re.compile(r'(?i)(?<!\w)(?:' + _SPACED_PATTERN + r')\s+(?:все\s+|all\s+)?(?:предыдущие|прежние|системные|previous|prior|system)\s+(?:инструкции|правила|указания|instructions|rules|prompts)[^.!?\n]*[.!?]?')),
    ('bracketed_role', re.compile(r'(?im)\[(?:system|developer|система|разработчик)\]\s*[^.!?\n]*[.!?]?')),
    ('spaced_override', re.compile(r'(?i)(?<![\w])и\s+г\s+н\s+о\s+р\s+и\s+р\s+у\s+й\s+(?:все\s+)?(?:предыдущие|прежние|системные)\s+(?:инструкции|правила|указания)[^.!?\n]*[.!?]?')),
    ('fake_role', re.compile(r'(?im)(?:^|(?<=\n))\s*(?:system|developer|система|разработчик)\s*[:：]\s*(?:игнорир|забуд|отмени|ответь|покажи|выведи|ignore|disregard|forget|answer|output|reveal|you\s+are\s+now|ты\s+теперь)[^.!?\n]*[.!?]?')),
    ('override', re.compile(r'(?i)(?:игнорируй(?:те)?|забудь(?:те)?|отмени(?:те)?)\s+(?:все\s+)?(?:предыдущие|прежние|системные|вышеуказанные)\s+(?:инструкции|правила|указания)[^.!?\n]*[.!?]?')),
    ('override_en', re.compile(r'(?i)\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|system|above)\s+(?:instructions|rules|prompts)[^.!?\n]*[.!?]?')),
    ('role_change', re.compile(r'(?i)(?:ты\s+теперь|теперь\s+ты|you\s+are\s+now)\s+(?:не\s+)?(?:ассистент|юрист|модель|chatgpt|an?\s+assistant)[^.!?\n]*[.!?]?')),
    ('secret', re.compile(r'(?i)(?:раскрой(?:те)?|покажи(?:те)?|выведи(?:те)?|отправь(?:те)?|reveal|print|expose)\s+(?:мне\s+)?(?:системн\w*\s+промпт\w*|api[- ]?ключ\w*|секрет\w*|system\s+prompt|api[- ]?key)[^.!?\n]*[.!?]?')),
    ('answer_override', re.compile(r'(?i)(?:ответь(?:те)?|напиши(?:те)?|выведи(?:те)?|reply|answer|output)\s+(?:только|строго|исключительно|only)\s+[«"\']?[^.!?\n]+[.!?]?')),
)

@dataclass(frozen=True)
class Span:
    start: int
    end: int
    reason: str
    confidence: str = 'high'


def find_spans(text: str) -> list[Span]:
    text = text or ''
    # Цитирование не делает метаинструкцию доверенной. Оригинал доступен отдельно.
    found = sorted((Span(m.start(), m.end(), reason) for reason, pattern in _RULES
                    for m in pattern.finditer(text)), key=lambda x: (x.start, -x.end))
    result = []
    for span in found:
        if result and span.start < result[-1].end:
            old = result[-1]
            result[-1] = Span(old.start, max(old.end, span.end), old.reason)
        else:
            result.append(span)
    return result


def clean(text: str) -> str:
    text = text or ''
    spans = find_spans(text)
    if not spans:
        return text
    parts = []
    cursor = 0
    for span in spans:
        parts.extend((text[cursor:span.start], MARKER))
        cursor = span.end
    parts.append(text[cursor:])
    return ''.join(parts)


def clean_value(value):
    if isinstance(value, str):
        return clean(value)
    if isinstance(value, list):
        return [clean_value(v) for v in value]
    if isinstance(value, dict):
        return {clean_value(k): clean_value(v) for k, v in value.items()}
    return value


def clean_dialog(dialog):
    return replace(dialog, raw=clean(dialog.raw), last_user=clean(dialog.last_user),
                   turns=[Turn(t.role, clean(t.text)) for t in dialog.turns],
                   history=[Turn(t.role, clean(t.text)) for t in dialog.history])
