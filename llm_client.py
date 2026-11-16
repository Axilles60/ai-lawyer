"""Клиент OpenAI-совместимого прокси: генерация, эмбеддинги, реранк (03-plan §4.5).

- Таймауты и ограниченные повторы с экспоненциальной паузой на 429/5xx/сеть (NFR-3, REG-13).
- Повтор при пустом content / finish_reason=length с увеличенным лимитом (Q8).
- Вырезание <think>…</think> из ответа (R6).
- Трейс каждого вызова: эндпоинт, модель, задержка, токены, статус — без ключей (P9.1).
- Дисковый кэш ответов только в локальной разработке (DEV_CACHE=1, P9.5).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from config import Settings, get_settings

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
STAT_KINDS = ("chat", "embed", "rerank")
MIN_ATTEMPT_S = 2.0  # меньше этого остатка новую попытку не начинаем


class LLMError(RuntimeError):
    """Ошибка API после всех повторов."""


def strip_think(text: str) -> str:
    text = THINK_RE.sub("", text or "")
    # Незакрытый блок рассуждения: отрезаем всё до последнего </think> или убираем открывающий тег
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.replace("<think>", "").strip()


def extract_json(text: str) -> Any:
    """Достаёт первый JSON-объект/массив из ответа модели (допускает ```json-обёртку и текст вокруг)."""
    text = strip_think(text)
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        depth, in_str, escape = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])
    raise ValueError("В ответе модели нет JSON")


@dataclass
class CallRecord:
    endpoint: str
    model: str
    ok: bool
    latency_s: float
    attempts: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached: bool = False
    note: str = ""


@dataclass
class Trace:
    """Журнал вызовов в рамках одного вопроса (передаётся явно, без глобального состояния)."""
    calls: list[CallRecord] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    def event(self, name: str, **data: Any) -> None:
        self.events.append({"t": round(time.time(), 3), "event": name, **data})

    def summary(self) -> dict:
        return {
            "llm_calls": sum(c.endpoint.startswith("chat") for c in self.calls),
            "latency_s": round(sum(c.latency_s for c in self.calls), 2),
            "prompt_tokens": sum(c.prompt_tokens for c in self.calls),
            "completion_tokens": sum(c.completion_tokens for c in self.calls),
            "failed_calls": sum(not c.ok for c in self.calls),
        }


@dataclass
class ChatResult:
    content: str
    finish_reason: str
    usage: dict
    raw_content: str


class LLMClient:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        if not self.s.base_url:
            raise LLMError("Не задан LLM_BASE_URL (или OPENAI_BASE_URL)")
        limits = httpx.Limits(max_connections=64, max_keepalive_connections=32)
        self._http = httpx.Client(timeout=self.s.http_timeout_s, limits=limits)
        self._cache_lock = threading.Lock()
        # Часы и пауза переопределяются в тестах (бюджеты времени проверяются без реального ожидания)
        self._clock = time.time
        self._sleep_fn = time.sleep
        self._stats_lock = threading.Lock()
        self.stats: dict[str, dict[str, int]] = {
            kind: {"calls": 0, "http_429": 0, "timeouts": 0, "budget_exhausted": 0} for kind in STAT_KINDS}
        if self.s.dev_cache:
            self.s.dev_cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- бюджет времени и счётчики (R2) ----------------

    def _bump(self, kind: str, name: str) -> None:
        with self._stats_lock:
            self.stats[kind][name] += 1

    def stats_snapshot(self) -> dict[str, dict[str, int]]:
        with self._stats_lock:
            return {kind: dict(values) for kind, values in self.stats.items()}

    def _safe_error(self, text: str) -> str:
        """Providers may echo credentials in an error. Redact before truncating or tracing."""
        secrets = {getattr(self.s, name, "") for name in ("llm_key", "emb_key", "rerank_key")}
        for secret in sorted((s for s in secrets if s), key=len, reverse=True):
            text = text.replace(secret, "[SECRET]")
        return re.sub(r"(?i)Bearer\s+[A-Za-z0-9._~+/-]+=*", "Bearer [SECRET]", text)

    def _limit(self, kind: str, deadline: float | None, bulk: bool = False) -> float:
        """Абсолютный предел времени логического вызова: бюджет типа вызова и внешний дедлайн — что раньше."""
        if bulk:
            budget = self.s.embed_index_budget_s
        else:
            budget = self.s.call_budget_chat_s if kind == "chat" else self.s.call_budget_aux_s
        limit = self._clock() + budget
        return limit if deadline is None else min(limit, deadline)

    # ---------------- низкий уровень ----------------

    def _cache_path(self, path: str, payload: dict) -> Path:
        key = hashlib.sha256(json.dumps([path, payload], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        return self.s.dev_cache_dir / key[:2] / f"{key}.json"

    def _post(self, path: str, key: str, payload: dict, trace: Trace | None, model: str,
              kind: str = "chat", limit: float | None = None, bulk: bool = False) -> dict:
        """Один вызов API с повторами. `limit` — абсолютный предел времени логического вызова (см. `_limit`).
        `bulk` — пачка эмбеддингов индекса: таймаут и бюджет из embed_index_* (на медленной платформе пачка из
        десятков текстов идёт заметно дольше одиночного запроса)."""
        cache_file = self._cache_path(path, payload) if self.s.dev_cache else None
        if cache_file and cache_file.is_file():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if trace is not None:
                trace.calls.append(CallRecord(path.strip("/"), model, True, 0.0, 0, cached=True))
            return data

        if not key:
            raise LLMError("Не задан API-ключ")
        if limit is None:
            limit = self._limit(kind, None, bulk)
        timeout_type = self.s.embed_index_timeout_s if bulk else \
            (self.s.chat_timeout_s if kind == "chat" else self.s.aux_timeout_s)
        attempt_cap = min(timeout_type, self.s.http_timeout_s)
        started = self._clock()
        last_error = ""
        attempts = 0
        self._bump(kind, "calls")
        for attempt in range(self.s.max_retries + 1):
            remaining = limit - self._clock()
            if remaining < MIN_ATTEMPT_S:
                self._bump(kind, "budget_exhausted")
                last_error = (last_error + "; " if last_error else "") + "бюджет времени вызова исчерпан"
                break
            attempts = attempt + 1
            retry_after = None
            try:
                response = self._http.post(
                    self.s.base_url + path,
                    headers={"Authorization": "Bearer " + key},
                    json=payload,
                    timeout=min(attempt_cap, remaining),
                )
                if response.status_code in RETRYABLE_STATUS:
                    last_error = f"HTTP {response.status_code}"
                    if response.status_code == 429:
                        self._bump(kind, "http_429")
                    retry_after = response.headers.get("retry-after")
                else:
                    if response.status_code >= 400:
                        last_error = f"HTTP {response.status_code}: {self._safe_error(response.text)[:200]}"
                        break
                    data = response.json()
                    if isinstance(data, dict) and data.get("error"):
                        last_error = f"API error: {self._safe_error(str(data['error']))[:200]}"
                    else:
                        if trace is not None:
                            usage = data.get("usage") or {}
                            details = usage.get("completion_tokens_details") or {}
                            trace.calls.append(CallRecord(
                                path.strip("/"), model, True, round(self._clock() - started, 3), attempts,
                                int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0),
                                int(details.get("reasoning_tokens") or 0)))
                        if cache_file:
                            with self._cache_lock:
                                cache_file.parent.mkdir(parents=True, exist_ok=True)
                                cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                        return data
            except (httpx.TimeoutException, httpx.TransportError, json.JSONDecodeError) as exc:
                last_error = type(exc).__name__
                if isinstance(exc, httpx.TimeoutException):
                    self._bump(kind, "timeouts")
            if attempt >= self.s.max_retries or not self._backoff(attempt, retry_after, limit, kind):
                break
        if trace is not None:
            trace.calls.append(CallRecord(path.strip("/"), model, False, round(self._clock() - started, 3),
                                          attempts, note=last_error))
        raise LLMError(f"{path}: {last_error}")

    def _backoff(self, attempt: int, retry_after: str | None, limit: float, kind: str) -> bool:
        """Пауза перед повтором. Retry-After учитывается, только если укладывается в остаток бюджета.
        False — на паузу и следующую попытку времени не хватает (повтор не делаем, паузу не ждём)."""
        remaining = limit - self._clock()
        delay = min(self.s.backoff_cap_s, 1.5 * (2 ** attempt)) + random.uniform(0, 0.5)
        if retry_after:
            try:
                wait = float(retry_after)
                if 0 <= wait <= remaining - MIN_ATTEMPT_S:
                    delay = max(delay, wait)
            except ValueError:
                pass
        if remaining - delay < MIN_ATTEMPT_S:
            self._bump(kind, "budget_exhausted")
            return False
        self._sleep_fn(delay)
        return True

    # ---------------- генерация ----------------

    def chat(self, messages: list[dict], *, max_tokens: int | None = None, temperature: float | None = None,
             trace: Trace | None = None, deadline: float | None = None) -> ChatResult:
        """`deadline` — абсолютное время (time.time()), позже которого вызов не продолжается (R2)."""
        return self._chat(messages, max_tokens, temperature, trace, self._limit("chat", deadline), deadline)

    def _chat(self, messages: list[dict], max_tokens: int | None, temperature: float | None,
              trace: Trace | None, time_limit: float, deadline: float | None = None) -> ChatResult:
        limit = max_tokens or self.s.max_tokens
        base_limit = limit
        temp = self.s.temperature if temperature is None else temperature
        for attempt in range(2):
            payload = {"model": self.s.llm_model, "messages": messages, "temperature": temp, "max_tokens": limit}
            data = self._post("/chat/completions", self.s.llm_key, payload, trace, self.s.llm_model,
                              "chat", time_limit)
            try:
                choice = data["choices"][0]
                raw = choice["message"].get("content") or ""
                finish = choice.get("finish_reason") or ""
            except (KeyError, IndexError, TypeError) as exc:
                raise LLMError(f"Неожиданный формат ответа: {type(exc).__name__}") from exc
            content = strip_think(raw)
            if content:
                return ChatResult(content, finish, data.get("usage") or {}, raw)
            if attempt == 1:
                if trace is not None:
                    trace.event("empty_content_exhausted", finish_reason=finish, max_tokens=limit)
                break
            # Пустой ответ: вероятно, reasoning съел лимит — повтор с бо́льшим лимитом (Q8).
            # T3.26: reasoning-модели с удвоенным лимитом токенов нуждаются в пропорционально
            # большем пределе времени вызова; внешний дедлайн вопроса — граница (H1, коллапс 154).
            limit = min(32768, limit * 2)
            factor = min(4.0, limit / base_limit)
            extended = self._clock() + self.s.call_budget_chat_s * factor
            time_limit = extended if deadline is None else min(extended, deadline)
            if trace is not None:
                trace.event("empty_content_retry", finish_reason=finish, new_limit=limit)
        raise LLMError("Модель вернула пустой ответ")

    def chat_json(self, messages: list[dict], *, max_tokens: int | None = None, temperature: float | None = None,
                  trace: Trace | None = None, attempts: int = 2, deadline: float | None = None) -> Any:
        """Генерация с разбором JSON; при невалидном JSON — повтор с напоминанием о формате.
        Базовый предел общий (R2); T3.26 расширяет его при повторе пустого ответа,
        оставаясь внутри внешнего дедлайна."""
        time_limit = self._limit("chat", deadline)
        last_error = ""
        msgs = list(messages)
        for _ in range(attempts):
            result = self._chat(msgs, max_tokens, temperature, trace, time_limit, deadline)
            try:
                return extract_json(result.content)
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                msgs = list(messages) + [
                    {"role": "assistant", "content": result.content[:4000]},
                    {"role": "user", "content": "Ответ не является валидным JSON. Верни только JSON по заданной схеме."},
                ]
        raise LLMError(f"Невалидный JSON после повторов: {last_error}")

    # ---------------- эмбеддинги ----------------

    def embed_bulk(self, texts: list[str], *, batch_size: int = 64, trace: Trace | None = None,
                   deadline: float | None = None) -> np.ndarray:
        """Эмбеддинги для индекса: то же, что embed, но с таймаутом и бюджетом пачки (embed_index_*)."""
        return self.embed(texts, batch_size=batch_size, trace=trace, deadline=deadline, bulk=True)

    def embed(self, texts: list[str], *, batch_size: int = 64, trace: Trace | None = None,
              deadline: float | None = None, bulk: bool = False) -> np.ndarray:
        """L2-нормированные эмбеддинги, float32, форма (len(texts), dim)."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        out: list[np.ndarray | None] = [None] * len(texts)
        for start in range(0, len(texts), batch_size):
            batch = [t if t.strip() else " " for t in texts[start:start + batch_size]]
            payload = {"model": self.s.emb_model, "input": batch, "encoding_format": "float"}
            data = self._post("/embeddings", self.s.emb_key, payload, trace, self.s.emb_model,
                              "embed", self._limit("embed", deadline, bulk), bulk)  # предел — на каждый батч
            items = data.get("data") or []
            if len(items) != len(batch):
                raise LLMError(f"Эмбеддер вернул {len(items)} векторов вместо {len(batch)}")
            for pos, item in enumerate(items):
                idx = int(item.get("index", pos))
                vec = np.asarray(item["embedding"], dtype=np.float32)
                norm = float(np.linalg.norm(vec)) or 1.0
                out[start + idx] = vec / norm
        return np.vstack(out)  # type: ignore[arg-type]

    # ---------------- реранк ----------------

    def rerank(self, query: str, documents: list[str], *, top_n: int | None = None,
               trace: Trace | None = None, deadline: float | None = None) -> list[tuple[int, float]]:
        """Список (индекс документа, relevance_score) по убыванию релевантности."""
        if not documents:
            return []
        payload = {"model": self.s.rerank_model, "query": query, "documents": documents,
                   "top_n": min(top_n or len(documents), len(documents))}
        data = self._post("/rerank", self.s.rerank_key, payload, trace, self.s.rerank_model,
                          "rerank", self._limit("rerank", deadline))
        items = data.get("results") or data.get("data") or []
        ranked = []
        for item in items:
            idx = int(item["index"])
            score = item.get("relevance_score", item.get("score", 0.0))
            if 0 <= idx < len(documents):
                ranked.append((idx, float(score) if score is not None and not math.isnan(float(score)) else 0.0))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked


def log(msg: str) -> None:
    """Диагностика в stderr (платформа показывает stderr в логах)."""
    print(msg, file=sys.stderr, flush=True)
