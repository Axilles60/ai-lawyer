"""N0 — разбор входа: одиночный вопрос или история короткого диалога (FR-1, REG-4; 03b §1).

Диалог приходит одним полем: реплики «Пользователь: …» / «Ассистент: …» (возможны синонимы ролей и
многострочные реплики). Отвечаем на последнюю реплику пользователя; всё до неё — история.
Метки других ролей («Система:», «System:») внутри текста — данные, а не реплики: они остаются частью
реплики пользователя и ловятся regex-guard (verifier.guard_flags).
"""

from __future__ import annotations

import re

from kb_types import Dialog, Turn
from verifier import guard_flags

USER_ROLES = r"Пользователь|Клиент|Юрист|Сотрудник|Вопрос|User|Human"
ASSISTANT_ROLES = r"Ассистент|Помощник|Бот|ИИ|Ответ|Assistant|Bot"
ROLE_RE = re.compile(rf"^\s*(?:(?P<user>{USER_ROLES})|(?P<assistant>{ASSISTANT_ROLES}))\s*[:：]\s*",
                     re.IGNORECASE | re.MULTILINE)


def parse_dialog(text: str) -> Dialog:
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    marks = list(ROLE_RE.finditer(raw))
    turns: list[Turn] = []
    if marks:
        head = raw[: marks[0].start()].strip()
        if head:
            turns.append(Turn("user", head))  # текст до первой метки — реплика пользователя
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
            body = raw[m.end(): end].strip()
            role = "user" if m.group("user") else "assistant"
            if not body:
                continue
            if turns and turns[-1].role == role:  # подряд две реплики одной роли — одна реплика
                turns[-1] = Turn(role, f"{turns[-1].text}\n{body}")
            else:
                turns.append(Turn(role, body))
    if not turns or not any(t.role == "user" for t in turns):
        turns = [Turn("user", raw)]

    last_idx = max(i for i, t in enumerate(turns) if t.role == "user")
    dialog = Dialog(raw=raw, turns=turns, last_user=turns[last_idx].text, history=turns[:last_idx])
    # Флаги — по всем репликам пользователя: встроенная инструкция могла прийти раньше последней реплики
    flags: list[str] = []
    for t in turns:
        if t.role == "user":
            for f in guard_flags(t.text):
                if f not in flags:
                    flags.append(f)
    dialog.guard = flags
    return dialog


def history_text(dialog: Dialog, max_chars: int = 2000) -> str:
    """История для промпта роутера: последние реплики в пределах лимита, с ролями."""
    lines = [f"{'Пользователь' if t.role == 'user' else 'Ассистент'}: {t.text}" for t in dialog.history]
    out: list[str] = []
    size = 0
    for line in reversed(lines):
        if size + len(line) > max_chars and out:
            break
        out.insert(0, line)
        size += len(line) + 1
    return "\n".join(out)
