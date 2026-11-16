"""Точка входа сабмита (REG-7; T1.16). Вся логика ответа — в pipeline.answer (C3); здесь только оркестрация.

Порядок:
1. Читаем вопросы и сразу пишем полный submission.csv с запасными ответами (R8): файл валиден с первой секунды.
2. Строим индекс из базы в DATA_DIR (разбор + эмбеддинги в пределах своего бюджета времени, C1, Q4).
3. Отвечаем параллельно (до MAX_WORKERS потоков, вопросы берутся из очереди по одному); контроллер (R3) по
   скорости ответов и доле 429 повышает ступень экономии и уменьшает число потоков. Файл атомарно
   перезаписывается по мере готовности (tmp → os.replace).
4. У дедлайна (RUN_DEADLINE_S, по умолчанию 17 мин из 20): вопросы, не начатые к жёсткому сроку, получают ответ
   модели на последней ступени экономии с коротким бюджетом, если время ещё есть; остальным — извлекающий ответ
   без LLM. Финальная запись и немедленный выход: зависшие сетевые вызовы не задерживают завершение.
Сбой любого API не роняет прогон (C2): нет эмбеддера — только полнотекстовый поиск; нет реранкера — порядок RRF;
нет LLM — найденные положения базы. В stderr — ход и статистика, без текстов вопросов и ключей.
"""

from __future__ import annotations

import csv
import dataclasses
import os
import queue
import sys
import tempfile
import threading
import time
from collections import Counter, deque
from pathlib import Path

from config import get_settings
from indexer import build_index
from kb_loader import find_inputs, load_kb, load_questions
from llm_client import LLMClient, LLMError, log
from llm_rag import llm_only_settings
from pipeline import MAX_ECONOMY, answer
from verifier import TECHNICAL_ANSWER, finalize

OUTPUT = "submission.csv"
WRITE_EVERY_S = 15.0        # не чаще — атомарная перезапись файла
FINAL_RESERVE_S = 20.0      # запас до дедлайна на запасные ответы и финальную запись
TAIL_RESERVE_S = 8.0        # из него — на финальную запись после ответов «хвоста»
TAIL_MIN_S = 5.0            # меньше этого до конца «хвоста» — вопросы не начинаем
TAIL_QUESTION_BUDGET_S = 20.0
CONTROL_EVERY = 5           # контроллер срабатывает после каждых стольких готовых ответов
RECENT_ANSWERS = 10         # по скольким последним ответам считается среднее время вопроса
RATE_WINDOW_CALLS = 20      # окно вызовов для доли 429
RATE_LIMIT_SHARE = 0.20     # выше — число потоков уменьшается вдвое


class RunController:
    """Адаптивный бюджет прогона (03-plan §4.5; PLAN R3). `on_done` вызывает только главный поток; рабочие
    потоки лишь читают `economy` и `active`.

    После каждых CONTROL_EVERY готовых ответов:
    - прогноз окончания = сейчас + осталось × среднее_время_последних_10 / активных_потоков; позже цели
      (старт + target_finish_share × platform_limit_s) → ступень экономии +1 (не выше MAX_ECONOMY, назад не
      понижается; окно времён очищается, чтобы следующая оценка видела уже новую ступень);
    - доля 429 среди последних RATE_WINDOW_CALLS вызовов API выше RATE_LIMIT_SHARE → активных потоков вдвое
      меньше (не ниже workers_min); окно отсчитывается заново от этого момента.
    Доля 429 — число ответов 429 на вызов API (повторы внутри вызова считаются), поэтому может превышать 1."""

    def __init__(self, settings, total: int, started: float, workers: int):
        self.s = settings
        self.total = total
        self.started = started
        self.economy = 0
        self.active = workers
        self.min_active = workers
        self.done = 0
        self.durations: deque[float] = deque(maxlen=RECENT_ANSWERS)
        self._snapshots: list[tuple[int, int]] = [(0, 0)]   # (вызовов API, ответов 429) после каждого ответа
        self._base = (0, 0)
        self.events: list[str] = []

    @property
    def target_time(self) -> float:
        return self.started + self.s.target_finish_share * self.s.platform_limit_s

    def forecast(self, now: float) -> float | None:
        if not self.durations:
            return None
        mean = sum(self.durations) / len(self.durations)
        return now + (self.total - self.done) * mean / max(1, self.active)

    def on_done(self, duration: float, calls: int = 0, http_429: int = 0, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self.done += 1
        self.durations.append(duration)
        self._snapshots.append((calls, http_429))
        if self.done % CONTROL_EVERY == 0:
            self._adjust(now, calls, http_429)

    def _window_start(self, calls: int) -> tuple[int, int]:
        start = self._snapshots[0]
        for snap in self._snapshots:
            if snap[0] <= calls - RATE_WINDOW_CALLS:
                start = snap
            else:
                break
        return max(start, self._base)

    def _adjust(self, now: float, calls: int, http_429: int) -> None:
        base_calls, base_429 = self._window_start(calls)
        window = calls - base_calls
        if window >= RATE_WINDOW_CALLS and (http_429 - base_429) / window > RATE_LIMIT_SHARE \
                and self.active > self.s.workers_min:
            self.active = max(self.s.workers_min, self.active // 2)
            self.min_active = min(self.min_active, self.active)
            self._base = (calls, http_429)
            self.events.append(f"429: потоков {self.active}")
        forecast = self.forecast(now)
        if forecast is not None and forecast > self.target_time and self.economy < MAX_ECONOMY:
            self.economy += 1
            self.durations.clear()
            self.events.append(f"прогноз {forecast - self.started:.0f} с: ступень {self.economy}")


def client_totals(client) -> tuple[int, int]:
    """(вызовов API, ответов 429) по всем типам вызовов; без клиента — нули."""
    snapshot = getattr(client, "stats_snapshot", None)
    if snapshot is None:
        return 0, 0
    stats = snapshot()
    return sum(v["calls"] for v in stats.values()), sum(v["http_429"] for v in stats.values())


# Интегрированные кандидаты: выключены по умолчанию, включаются через Settings/окружение.
LLM_ONLY_DEFAULT = False
CRITIC_DEFAULT = False  # интеграционный профиль: проверяющий проход только по флагу


def _enabled(name: str, default: bool) -> bool:
    flag = os.environ.get(name, "").strip().lower()
    if flag in {"0", "false", "no", "off", "нет"}:
        return False
    return default


def llm_only_enabled(settings) -> bool:
    return _enabled("USE_LLM_ONLY", LLM_ONLY_DEFAULT or bool(settings.use_llm_only))


def runtime_settings(settings):
    """Единая подготовка явно включённых кандидатов для пакетного и веб-режима."""
    settings = dataclasses.replace(settings, use_critic=_enabled("USE_CRITIC", settings.use_critic),
                                   use_llm_only=llm_only_enabled(settings))
    return llm_only_settings(settings) if settings.use_llm_only else settings


def write_submission(path: Path, ids: list[str], answers: dict[str, str]) -> None:
    """Атомарная запись: весь файл во временный рядом, затем os.replace (R8)."""
    directory = path.parent if str(path.parent) else Path(".")
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".submission_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
            writer.writerow(["id", "answer"])
            for qid in ids:
                writer.writerow([qid, finalize(answers.get(qid) or TECHNICAL_ANSWER) or TECHNICAL_ANSWER])
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def make_client(settings):
    if not settings.base_url or not settings.llm_key:
        log("[solution] нет LLM_BASE_URL / LLM_API_KEY — работаем без LLM (извлекающие ответы)")
        return None
    try:
        return LLMClient(settings)
    except LLMError as exc:
        log(f"[solution] клиент API не создан: {exc}")
        return None


def run(output: Path = Path(OUTPUT)) -> dict:
    started = time.time()
    s = runtime_settings(get_settings())
    if s.use_llm_only:
        log("[solution] режим LLM-only: поиск и ранжирование силами модели, эмбеддер и реранкер не вызываются")
    deadline = started + s.run_deadline_s
    log(f"[solution] старт; параметры: {s.describe()}")

    questions_path, kb_path = find_inputs(s.data_dir)
    questions = load_questions(questions_path)
    ids = [qid for qid, _ in questions]
    if len(set(ids)) != len(ids):
        log("[solution] в questions.csv повторяются ID — для каждого берётся последний вопрос")
        ids = list(dict.fromkeys(ids))
    texts = dict(questions)
    answers: dict[str, str] = {qid: TECHNICAL_ANSWER for qid in ids}
    write_submission(output, ids, answers)  # файл валиден с первой секунды
    log(f"[solution] вопросов: {len(ids)}; запасной submission.csv записан")

    client = make_client(s)
    raw = load_kb(kb_path)
    index_deadline = started + s.run_deadline_s * s.index_share
    # LLM-only: векторный индекс не строится (эмбеддер не вызывается), готовые векторы архива не используются
    index = build_index(raw, None if s.use_llm_only else client, s, deadline=index_deadline)
    log("[solution] " + index.report_text().replace("\n", "\n[solution] "))

    ctl = RunController(s, len(ids), started, s.max_workers)
    tail_settings = dataclasses.replace(s, question_budget_s=TAIL_QUESTION_BUDGET_S)
    tail = threading.Event()      # «хвост»: вопросы, не начатые к жёсткому сроку
    stop = threading.Event()
    todo: queue.Queue = queue.Queue()
    for qid in ids:
        todo.put(qid)
    results: queue.Queue = queue.Queue()

    def work(worker_no: int) -> None:
        """Вопросы берутся из очереди по одному: новая ступень экономии действует на ещё не начатые."""
        while not stop.is_set():
            if worker_no >= ctl.active:      # контроллер уменьшил число потоков
                time.sleep(0.2)
                continue
            try:
                qid = todo.get_nowait()
            except queue.Empty:
                return
            if tail.is_set():
                economy, settings_q, deadline_q = MAX_ECONOMY, tail_settings, deadline - TAIL_RESERVE_S
            else:
                economy, settings_q, deadline_q = ctl.economy, s, deadline - FINAL_RESERVE_S
            t0 = time.time()
            try:
                res = answer(texts[qid], index, client, settings_q, deadline=deadline_q, economy=economy)
                if res.fallback:   # причина запасного ответа: типы событий трейса и текст ошибки, без текста вопроса
                    why = [f"{e['event']}: {str(e.get('error', ''))[:100]}" for e in res.trace.events
                           if e["event"] in ("error", "generation_failed", "light_failed", "regenerate_failed")]
                    log(f"[solution] запасной ответ {qid} ({res.mode}): {'; '.join(why) or 'проверки V1/V2/V10'}")
                item = (qid, res.answer, res.fallback, res.mode, res.decision, economy, time.time() - t0)
            except Exception as exc:  # answer не бросает, но страхуемся
                log(f"[solution] вопрос {qid}: {type(exc).__name__}")
                item = (qid, TECHNICAL_ANSWER, True, "technical", "answer", economy, time.time() - t0)
            results.put(item)

    for n in range(s.max_workers):
        threading.Thread(target=work, args=(n,), daemon=True, name=f"worker-{n}").start()

    done: set[str] = set()
    fallbacks = 0
    modes: Counter = Counter()
    economies: Counter = Counter()
    decisions: Counter = Counter()
    durations: list[float] = []   # время каждого вопроса, с (для отчёта прогона)
    last_write = time.time()
    phase_end = deadline - FINAL_RESERVE_S
    while len(done) < len(ids):
        now = time.time()
        if now >= phase_end:
            tail_end = deadline - TAIL_RESERVE_S
            if not tail.is_set() and client is not None and not todo.empty() and tail_end - now >= TAIL_MIN_S:
                log(f"[solution] жёсткий срок: {todo.qsize()} не начатых вопросов — ответ на ступени "
                    f"{MAX_ECONOMY} с бюджетом {TAIL_QUESTION_BUDGET_S:.0f} с")
                tail.set()
                phase_end = tail_end
                continue
            break
        try:
            qid, text, fb, mode, decision, economy, duration = results.get(
                timeout=min(phase_end - now, WRITE_EVERY_S))
        except queue.Empty:
            qid = None
        if qid is not None:
            answers[qid] = text
            done.add(qid)
            fallbacks += fb
            modes[mode] += 1
            economies[economy] += 1
            decisions[decision] += 1
            durations.append(duration)
            ctl.on_done(duration, *client_totals(client))
            if len(done) % 10 == 0 or len(done) == len(ids):
                log(f"[solution] готово {len(done)}/{len(ids)}, запасных {fallbacks}, ступень {ctl.economy}, "
                    f"потоков {ctl.active}, {time.time() - started:.0f} с")
        if time.time() - last_write >= WRITE_EVERY_S or len(done) == len(ids):
            write_submission(output, ids, answers)
            last_write = time.time()
    stop.set()
    for event in ctl.events:
        log(f"[solution] контроллер: {event}")

    unfinished = [qid for qid in ids if qid not in done]
    if unfinished:
        log(f"[solution] дедлайн: не успели {len(unfinished)} — извлекающие ответы без LLM")
        for qid in unfinished:
            if time.time() > deadline:
                break
            try:
                answers[qid] = answer(texts[qid], index, None, s, deadline=deadline).answer
            except Exception:
                answers[qid] = TECHNICAL_ANSWER
            fallbacks += 1
            modes["extractive"] += 1
    write_submission(output, ids, answers)
    calls, http_429 = client_totals(client)
    stats = {"questions": len(ids), "answered": len(done), "fallbacks": fallbacks, "unfinished": len(unfinished),
             "seconds": round(time.time() - started, 1),
             "modes": dict(modes), "economy": dict(sorted(economies.items())),
             "economy_max": max(economies) if economies else 0,
             "clarify": decisions["clarify"], "refuse": decisions["refuse"],
             "http_429": http_429, "api_calls": calls, "min_workers": ctl.min_active,
             "sec_per_question": round(sum(durations) / len(durations), 1) if durations else 0.0,
             "max_question_s": round(max(durations), 1) if durations else 0.0}
    log(f"[solution] итог: {stats}")
    return stats


def main() -> None:
    try:
        stats = run()
    except Exception as exc:
        # Последний рубеж: даже при сбое чтения базы файл с ответами на все ID должен остаться (C2)
        log(f"[solution] критическая ошибка: {type(exc).__name__}: {str(exc)[:200]}")
        try:
            questions_path, _ = find_inputs(get_settings().data_dir)
            ids = [qid for qid, _ in load_questions(questions_path)]
            if not Path(OUTPUT).is_file():
                write_submission(Path(OUTPUT), ids, {})
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    # Немедленный выход: потоки с зависшими сетевыми вызовами не должны держать процесс после записи файла
    os._exit(0 if stats["questions"] else 1)


if __name__ == "__main__":
    main()
