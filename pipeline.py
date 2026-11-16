"""Единое ядро ответа (C3; 03-plan §4.2; 06 §5): pipeline.answer — его вызывают сабмит, веб-чат и оценка.

Узлы среза:
  N0 разбор диалога → N1 regex-guard → N2 роутер (LLM #1) → N3 разрешение (документы, редакции, уточнение)
  → N4–N5 поиск, реранк, бюджет → N7 генерация (LLM #2) → N8 проверки V1/V2/V10
  → при замечаниях одна повторная генерация с ними → удаление неподтверждённых утверждений
  → N10 сборка текста и ссылок кодом; при отсутствии содержательного ответа — извлекающий запасной ответ.

answer никогда не бросает исключение наружу (06 §5): любой сбой → Result с запасным ответом и fallback=True.
Состояние между вопросами не переносится (REG-4).
"""

from __future__ import annotations

import time
import json
import re

from dialog import parse_dialog
from generator import generate
from dossier import light_retrieval
from critic import critique
from kb_types import Draft, Result
from llm_rag import effective_budget, note_generation_failure, retrieve_llm
from llm_client import LLMError, Trace
from resolver import resolve_request
from retriever import retrieve
from router import route
from verifier import (MANIPULATION_ANSWER, OFFTOPIC_ANSWER, TECHNICAL_ANSWER, check, extractive_fallback,
                      finalize, issues_note, prune, render)

REGEN_MIN_SECONDS = 20.0  # если до дедлайна меньше — повторной генерации нет
LIGHT_MIN_SECONDS = 5.0   # если до дедлайна меньше — облегчённого повтора генерации нет
MAX_ECONOMY = 4           # лестница деградации (03-plan §4.5): 0 полный путь … 4 только FTS


class _EconomyClient:
    """Клиент с отключёнными дорогими шагами: поиск и так переживает LLMError (реранк → порядок RRF,
    эмбеддинг запроса → только FTS), поэтому ступени 3–4 реализованы отказом клиента, а не правкой поиска."""

    def __init__(self, client, rerank: bool, embed: bool):
        self._client, self._rerank, self._embed = client, rerank, embed

    def chat_json(self, *args, **kwargs):
        return self._client.chat_json(*args, **kwargs)

    def chat(self, *args, **kwargs):
        return self._client.chat(*args, **kwargs)

    def rerank(self, *args, **kwargs):
        if not self._rerank:
            raise LLMError("реранк отключён ступенью экономии")
        return self._client.rerank(*args, **kwargs)

    def embed(self, *args, **kwargs):
        if not self._embed:
            raise LLMError("эмбеддинг отключён ступенью экономии")
        return self._client.embed(*args, **kwargs)


def economy_client(client, economy: int):
    """Ступень 3 — без реранка; ступень 4 — ещё и без эмбеддинга запроса. Генерация остаётся всегда."""
    if client is None or economy < 3:
        return client
    return _EconomyClient(client, rerank=False, embed=economy < 4)


def _time_left(deadline: float | None) -> float:
    return float("inf") if deadline is None else deadline - time.time()


def fallback_node(retrieval, index) -> str | None:
    """Узел для извлекающего запасного ответа: первый найденный пункт с содержательным положением — не заголовок
    раздела, не преамбула и не реквизиты (их поднимают совпадения по названию контрагента)."""
    if retrieval is None:
        return None
    ranked = [n for n, _ in retrieval.reranked] + [c.node_id for c in retrieval.candidates]
    for nid in ranked:
        node = index.node(nid)
        if node.path == "преамбула" or node.label.startswith("раздел") or len(node.text) < 40:
            continue
        if node.heading and len(node.text) <= len(node.heading) + 10:
            continue
        return nid
    return retrieval.best_node_id


def _cited_nodes(draft: Draft, ctx) -> list[str]:
    out = []
    for c in draft.claims:
        for t in c.sources:
            target = ctx.tag_to_node.get(t)
            if target and target not in out:
                out.append(target)
    return out


_SANCTION_RE = re.compile(r"штраф|санкци|неустойк|ответственност|пен[ияейю]|взыскан", re.IGNORECASE)


def sanction_review(draft, ctx, index, question, client, trace, deadline):
    """Один выборочный вызов; при сбое возвращает пустое замечание."""
    from prompts import SANCTION_REVIEW_SYSTEM
    selected = [c for c in draft.claims if _SANCTION_RE.search(c.text)]
    if not selected:
        return ""
    entries = []
    for claim in selected:
        sources = []
        for tag in claim.sources:
            target = ctx.tag_to_node.get(tag)
            if not target:
                return ""
            sources.append(index.node(target).text if "." in tag else index.doc_text(target))
        if not sources or sum(map(len, sources)) > 16000:
            return ""
        entries.append({"claim": claim.text, "sources": sources})
    review_input = {"question": question, "claims": entries}
    if getattr(index, "use_safe_context", False):
        from safe_context import clean_value
        review_input = clean_value(review_input)
    payload = json.dumps(review_input, ensure_ascii=False)
    try:
        data = client.chat_json([{"role": "system", "content": SANCTION_REVIEW_SYSTEM},
                                 {"role": "user", "content": payload}], trace=trace,
                                deadline=min(deadline, time.time() + 25), attempts=1)
        if not isinstance(data, dict) or data.get("needs_fix") is not True:
            return ""
        note = data.get("note")
        return note.strip()[:600] if isinstance(note, str) else ""
    except (LLMError, ValueError, TypeError, KeyError):
        return ""

CRITIC_MIN_SECONDS = 45.0  # проверяющий проход: меньше этого до дедлайна — не запускаем
CYCLE_MIN_SECONDS = 60.0  # режим LLM-only: меньше этого до дедлайна — новый цикл RAG не начинаем


def _result_rank(result: Result) -> int:
    """0 — содержательный ответ, 1 — обоснованный отказ, 2 — запасной ответ (чем меньше, тем лучше)."""
    if result.fallback:
        return 2
    return 1 if result.decision == "refuse" else 0


def _critic_refine(index, dialog, router, resolution, retrieval, client, settings, trace, deadline, draft, ctx,
                   extra_note: str):
    """Проверяющий проход (critic.py): замечания → перегенерация с ними. Новый ответ принимается, только если проходит
    проверки V1/V2/V10 (после обрезки неподтверждённого остаётся прямой ответ) и остаётся решением answer."""
    for round_no in range(1, int(getattr(settings, "critic_rounds", 1)) + 1):
        if _time_left(deadline) < CRITIC_MIN_SECONDS:
            break
        try:
            verdict = critique(index, dialog, router, draft, ctx, client, trace, deadline)
        except (LLMError, ValueError) as exc:
            trace.event("critic_failed", error=str(exc)[:200])
            break
        trace.event("critic", round=round_no, ok=verdict.ok, problems=len(verdict.problems),
                    missing=len(verdict.missing_tags))
        if verdict.ok:
            break
        note = (extra_note + "\n" if extra_note else "") + verdict.note()
        try:
            cand, cand_ctx = generate(index, dialog, router, resolution, retrieval, client, settings, trace,
                                      extra_note=note, deadline=deadline)
        except (LLMError, ValueError) as exc:
            trace.event("critic_regen_failed", error=str(exc)[:200])
            break
        issues = check(cand, cand_ctx, index, settings)
        if issues:
            if any(i.claim_index is None or i.claim_index == 0 for i in issues):
                trace.event("critic_rejected", reason="issues")
                break
            cand = prune(cand, issues)
        if cand.decision != "answer" or not cand.claims:
            trace.event("critic_rejected", reason="decision")
            break
        draft, ctx = cand, cand_ctx
        trace.event("critic_applied", round=round_no)
    return draft, ctx


def _full_draft_evidence(candidates, question, index):
    """Complete primary evidence, deduplicated across candidate-local context tags."""
    if index is None or not candidates:
        raise ValueError("missing_index")
    source_ids, sources, drafts = {}, [], []
    for number, (draft, context, issues) in enumerate(candidates, 1):
        tags = set()
        for claim in draft.claims:
            tags.update(claim.sources)
            tags.update(re.findall(r"\[\[(Д\d+(?:\.\d+)*)\]\]", claim.text))
        tags.update(re.findall(r"\[\[(Д\d+(?:\.\d+)*)\]\]", draft.missing))
        mapping = {}
        for tag in sorted(tags):
            target = context.tag_to_node.get(tag)
            if not isinstance(target, str) or not target:
                raise ValueError("unknown_source_tag")
            kind = "node" if "." in tag else "document"
            key = (kind, target)
            if key not in source_ids:
                try:
                    node = index.node(target) if kind == "node" else None
                    text = node.text if node is not None else index.doc_text(target)
                    card = index.doc(node.doc_id if node is not None else target)
                except (KeyError, AttributeError, TypeError):
                    raise ValueError("unavailable_primary_source") from None
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("unavailable_primary_source")
                source_id = f"S{len(sources) + 1}"
                source_ids[key] = source_id
                sources.append({"id": source_id, "kind": kind, "target": target, "text": text,
                                "label": node.label if node is not None else "document",
                                "path": node.path if node is not None else "",
                                "document": {"id": card.doc_id, "file": card.file, "act": card.act,
                                             "number": card.number, "title": card.title, "date": card.date,
                                             "valid_from": card.valid_from, "valid_to": card.valid_to,
                                             "export_date": card.export_date}})
            mapping[tag] = source_ids[key]
        drafts.append({"id": number, "decision": draft.decision, "missing": draft.missing,
                       "issues": [{"check": item.check, "claim_index": item.claim_index,
                                   "message": item.message} for item in issues],
                       "claims": [{"text": claim.text, "sources": list(claim.sources),
                                   "quote": claim.quote} for claim in draft.claims],
                       "tag_to_source": mapping})
    payload = {"question": question, "drafts": drafts, "primary_sources": sources}
    if getattr(index, "use_safe_context", False):
        from safe_context import clean_value
        payload = clean_value(payload)
    return payload


def _judge_pick(candidates, question, client, trace, deadline, index=None, full_evidence=False):
    """T3.35: один evidence-судья выбирает лучшего кандидата по опоре на цитаты.
    Любой сбой/невалидный ответ → None (остаётся структурный выбор)."""
    from prompts import DRAFT_JUDGE_SYSTEM, DRAFT_JUDGE_FULL_SYSTEM
    import json as _json
    if full_evidence:
        try:
            payload = _json.dumps(_full_draft_evidence(candidates, question, index), ensure_ascii=False)
        except Exception:
            trace.event("draft_judge_skipped", reason="full_evidence_unavailable")
            return None
        if len(payload) > 24000:
            trace.event("draft_judge_skipped", reason="full_evidence_budget", chars=len(payload), budget=24000)
            return None
        system, candidate_count = DRAFT_JUDGE_FULL_SYSTEM, len(candidates)
    else:
        rows = []
        for n, (d, _ctx, issues) in enumerate(candidates[:6], 1):
            claims = "; ".join(f"{c.text} — цитата: «{c.quote[:120]}»" for c in d.claims[:4]) or "(без утверждений)"
            rows.append(f"ЧЕРНОВИК {n} (ошибок проверки: {len(issues)}): {claims}")
        payload = _json.dumps({"question": question, "drafts": rows}, ensure_ascii=False)
        system, candidate_count = DRAFT_JUDGE_SYSTEM, len(candidates[:6])
    try:
        data = client.chat_json([{"role": "system", "content": system},
                                 {"role": "user", "content": payload}], trace=trace,
                                **({"deadline": deadline} if deadline is not None else {}))
        if full_evidence:
            value = data.get("best") if isinstance(data, dict) else None
            best = value if type(value) is int else None
        else:
            best = int(data.get("best")) if isinstance(data, dict) else None
    except Exception as exc:
        trace.event("draft_judge_failed", error=f"{type(exc).__name__}: {exc}"[:200])
        return None
    if best is None or not 1 <= best <= candidate_count:
        trace.event("draft_judge_invalid", answer=str(best)[:40])
        return None
    trace.event("draft_judge", picked=best, of=candidate_count)
    return best - 1


def _generate_stage(index, dialog, router, resolution, retrieval, client, settings, trace, deadline, economy,
                    extra_note: str = "") -> Result:
    """Генерация → проверки V1/V2/V10 → повторная генерация → сборка текста (03b §4–5). Сбой модели даёт облегчённый
    повтор, и только затем запасной ответ."""
    q19 = getattr(settings, "use_answer_variants", False)
    llm_only = getattr(settings, "use_llm_only", False)
    ctx = None
    mode = "full"
    draft = None
    try:
        draft, ctx = generate(index, dialog, router, resolution, retrieval, client, settings, trace,
                              extra_note=extra_note, deadline=deadline)
    except (LLMError, ValueError) as exc:
        trace.event("generation_failed", error=str(exc)[:200])
        if llm_only:
            note_generation_failure(str(exc), settings)
    # Облегчённый повтор (03b §4.4): урезанный контекст, без повторной генерации после проверок.
    # В режиме LLM-only контекст большой, поэтому уменьшений два (1/2 и 1/6 бюджета) и общий потолок запоминается.
    for fraction in ((2, 6) if llm_only else (2,)):
        if draft is not None or _time_left(deadline) <= LIGHT_MIN_SECONDS:
            break
        mode = "light"
        trace.event("mode", mode=mode)
        budget = settings.context_budget_chars // fraction
        if llm_only:
            budget = min(budget, effective_budget(settings))
        light = light_retrieval(retrieval, budget)
        try:
            draft, ctx = generate(index, dialog, router, resolution, light, client, settings, trace,
                                  extra_note=extra_note, deadline=deadline)
        except (LLMError, ValueError) as exc:
            trace.event("light_failed", error=str(exc)[:200])
            if llm_only:
                note_generation_failure(str(exc), settings)
    if draft is None:
        # Модель недоступна совсем: только теперь — извлекающий ответ
        best = fallback_node(retrieval, index)
        trace.event("mode", mode="extractive" if best else "technical")
        return Result(answer=extractive_fallback(ctx, index, best), decision="answer",
                      node_ids=[best] if best else [], fallback=True, trace=trace,
                      mode="extractive" if best else "technical")
    issues = check(draft, ctx, index, settings)
    # T3.27/T3.32 (OFF): независимые генерации и выбор лучшего черновика; до best_of_n
    # доп. генераций, пока есть время — полное использование бюджета платформы.
    # T3.35 (OFF): финальный выбор одним evidence-судьёй по опоре утверждений на цитаты;
    # при сбое судьи — структурный скор (issues↓, при равенстве — claims↑).
    if (getattr(settings, "use_best_of_two", False) and client is not None and mode == "full"
            and economy == 0):
        score = lambda d, i: (-len(i), len(d.claims))  # noqa: E731
        candidates = [(draft, ctx, list(issues))]
        attempts = 0
        max_alts = min(getattr(settings, "best_of_n", 2), 6)
        while attempts < max_alts and _time_left(deadline) > 2 * REGEN_MIN_SECONDS:
            attempts += 1
            try:
                alt_draft, alt_ctx = generate(index, dialog, router, resolution, retrieval, client,
                                              settings, trace, extra_note=extra_note, deadline=deadline)
                alt_issues = check(alt_draft, alt_ctx, index, settings)
            except Exception as exc:  # любой сбой попытки — сохраняем лучший достигнутый
                trace.event("best_of_two_failed", error=f"{type(exc).__name__}: {exc}"[:200])
                break
            trace.event("best_of_two", primary=score(draft, issues), alt=score(alt_draft, alt_issues))
            candidates.append((alt_draft, alt_ctx, list(alt_issues)))
            if score(alt_draft, alt_issues) > score(draft, issues):
                draft, ctx, issues = alt_draft, alt_ctx, list(alt_issues)
        if (getattr(settings, "use_draft_judge", False) and len(candidates) > 1
                and _time_left(deadline) > REGEN_MIN_SECONDS):
            if getattr(settings, "use_draft_judge_full", False):
                picked = _judge_pick(candidates, getattr(dialog, "raw", None) or getattr(dialog, "last_user", ""),
                                     client, trace, deadline, index=index, full_evidence=True)
            else:
                picked = _judge_pick(candidates, getattr(dialog, "last_user", ""), client, trace, deadline)
            if picked is not None:
                draft, ctx, issues = candidates[picked]
    regenerations = 0
    max_regenerations = 0 if (mode == "light" or economy >= 1) else getattr(settings, "max_regenerations", 1)
    # T3.24: перегенерация работает от лучшего достигнутого черновика и не ухудшает его
    # (урок 8 разбора соревнований: ранние попытки перезаписывали лучший результат).
    best = (draft, ctx, list(issues))
    while best[2] and regenerations < max_regenerations and _time_left(deadline) > REGEN_MIN_SECONDS:
        regenerations += 1
        trace.event("regenerate", issues=[f"{i.check}:{i.claim_index}" for i in best[2]])
        try:
            cand, cand_ctx = generate(index, dialog, router, resolution, retrieval, client, settings, trace,
                                      extra_note=(extra_note + "\n" if extra_note else "")
                                      + issues_note(best[2], best[0]),
                                      deadline=deadline)
        except Exception as exc:  # любой сбой перегенерации — лучший черновик сохранён
            trace.event("regenerate_failed", error=f"{type(exc).__name__}: {exc}"[:200])
            break
        cand_issues = check(cand, cand_ctx, index, settings)
        if (-len(cand_issues), len(cand.claims)) > (-len(best[2]), len(best[0].claims)):
            best = (cand, cand_ctx, list(cand_issues))

    draft, ctx, issues = best
    all_issues = list(issues)
    if issues:
        first_bad = any(i.claim_index == 0 for i in issues)
        structural = any(i.claim_index is None for i in issues)
        draft = prune(draft, issues)
        trace.event("pruned", removed=len({i.claim_index for i in issues if i.claim_index is not None}))
        if draft.decision == "answer" and (first_bad or structural or not draft.claims):
            # Прямой ответ не подтверждён — честный запасной ответ вместо неподтверждённого текста
            return Result(answer=extractive_fallback(ctx, index, fallback_node(retrieval, index)), decision="answer",
                          claims=draft.claims, node_ids=_cited_nodes(draft, ctx), fallback=True,
                          issues=all_issues, trace=trace, mode="extractive", missing=draft.missing)

    if (getattr(settings, "use_sanction_condition_review", False) and client is not None
            and mode == "full" and economy == 0 and draft.decision == "answer"
            and _time_left(deadline) > 50):
        note = sanction_review(draft, ctx, index, dialog.raw, client, trace, deadline)
        if note and _time_left(deadline) > REGEN_MIN_SECONDS:
            trace.event("sanction_review_fix")
            try:
                revised, revised_ctx = generate(index, dialog, router, resolution, retrieval, client, settings,
                                                trace, extra_note=note, deadline=deadline)
                if revised.decision == "answer" and not check(revised, revised_ctx, index, settings):
                    draft, ctx = revised, revised_ctx
            except (LLMError, ValueError):
                trace.event("sanction_review_regeneration_failed")

    if q19 and draft.decision == "clarify":
        best = fallback_node(retrieval, index)
        trace.event("q19_invalid_decision", decision="clarify")
        return Result(answer=extractive_fallback(ctx, index, best), decision="answer", fallback=True,
                      issues=all_issues, trace=trace, node_ids=[best] if best else [],
                      mode="extractive" if best else "technical")
    if (getattr(settings, "use_critic", False) and economy == 0 and mode == "full" and draft.decision == "answer"
            and draft.claims):
        draft, ctx = _critic_refine(index, dialog, router, resolution, retrieval, client, settings, trace, deadline,
                                    draft, ctx, extra_note)
    text = render(draft, ctx, index)
    return Result(answer=text, decision=draft.decision, claims=draft.claims, node_ids=_cited_nodes(draft, ctx),
                  fallback=False, issues=all_issues, trace=trace, mode=mode, missing=draft.missing)


def _rag_cycles(index, dialog, router, resolution, client, settings, trace, deadline, first: Result) -> Result:
    """Повтор всего цикла RAG (поиск → ранжирование → досье → генерация) при отказе или запасном ответе: поиск идёт
    заново с подсказкой «что не нашлось», генерация получает замечание перечитать весь контекст. Берётся лучший итог."""
    best = first
    for cycle in range(1, int(getattr(settings, "llm_rag_cycles", 0)) + 1):
        if _result_rank(best) == 0 or _time_left(deadline) < CYCLE_MIN_SECONDS:
            break
        hint = (best.missing or "").strip()[:400]
        trace.event("rag_cycle", cycle=cycle, previous=_result_rank(best), hint=bool(hint))
        try:
            retrieval = retrieve_llm(index, dialog, router, resolution, client, settings, trace, deadline, 0, hint)
        except Exception as exc:  # цикл не должен ронять вопрос — остаётся прежний итог
            trace.event("rag_cycle_failed", error=f"{type(exc).__name__}: {str(exc)[:160]}")
            break
        note = ("Предыдущая попытка не дала подтверждённого ответа" + (f" (не нашлось: {hint})" if hint else "") +
                ". Перечитай весь контекст внимательно: нужное положение могло быть в другом документе или пункте. "
                "Если ответ действительно не следует из документов — верни refuse.")
        try:
            result = _generate_stage(index, dialog, router, resolution, retrieval, client, settings, trace, deadline,
                                     0, extra_note=note)
        except Exception as exc:
            trace.event("rag_cycle_failed", error=f"{type(exc).__name__}: {str(exc)[:160]}")
            break
        if _result_rank(result) < _result_rank(best):
            best = result
    return best


def answer(question: str, index, client, settings, deadline: float | None = None, economy: int = 0) -> Result:
    """`economy` — ступень лестницы деградации: 0 полный путь; 1 без повторной генерации после проверок;
    2 + роутер правилами вместо LLM; 3 + без реранка; 4 + без эмбеддинга запроса. Генерация моделью есть всегда.
    В режиме USE_LLM_ONLY поиск и ранжирование ведёт LLM (llm_rag.py), циклы RAG — только на ступени 0."""
    trace = Trace()
    economy = max(0, min(MAX_ECONOMY, int(economy)))
    if economy:
        trace.event("economy", level=economy)
    retrieval = None
    # Дедлайн вопроса (R2): не позже внешнего дедлайна и не дольше question_budget_s; уходит во все вызовы API
    question_deadline = time.time() + getattr(settings, "question_budget_s", 150.0)
    deadline = question_deadline if deadline is None else min(deadline, question_deadline)
    try:
        dialog = parse_dialog(question)
        if getattr(settings, "use_safe_context", False):
            from safe_context import clean_dialog
            dialog = clean_dialog(dialog)
        trace.event("dialog", turns=len(dialog.turns), guard=dialog.guard)

        router, rec = route(dialog, index, None if economy >= 2 else client, settings, trace, deadline)
        if getattr(settings, "use_safe_context", False):
            from safe_context import clean_value
            for field in ("query", "doc_refs", "node_refs", "entities", "user_claims", "subqueries", "stepback"):
                setattr(router, field, clean_value(getattr(router, field)))
        if router.scope == "offtopic":
            return Result(answer=OFFTOPIC_ANSWER, decision="refuse", trace=trace)
        if router.manipulation and not router.query:
            return Result(answer=MANIPULATION_ANSWER, decision="refuse", trace=trace)

        q19 = getattr(settings, "use_answer_variants", False)
        resolution = resolve_request(dialog, router, rec, index, use_answer_variants=True) if q19 else \
            resolve_request(dialog, router, rec, index)
        if q19:
            trace.event("resolution", candidates=resolution.candidates, primary_doc=resolution.primary_doc,
                        ambiguity_reason=resolution.ambiguity_reason, target_docs=resolution.target_docs,
                        source=resolution.source)
        if resolution.clarify and not q19:
            trace.event("clarify", candidates=len(resolution.candidates))
            return Result(answer=finalize(resolution.clarify), decision="clarify", trace=trace)

        llm_only = bool(getattr(settings, "use_llm_only", False)) and client is not None
        if llm_only:
            retrieval = retrieve_llm(index, dialog, router, resolution, client, settings, trace, deadline, economy)
        else:
            retrieval = retrieve(index, dialog, router, resolution, economy_client(client, economy), settings, trace,
                                 deadline)
        if client is None:
            # Без LLM генерации нет: отдаём найденное положение, а не заглушку (C2)
            best = fallback_node(retrieval, index)
            return Result(answer=extractive_fallback(None, index, best), decision="answer",
                          node_ids=[best] if best else [], fallback=True, trace=trace,
                          mode="extractive" if best else "technical")

        result = _generate_stage(index, dialog, router, resolution, retrieval, client, settings, trace, deadline,
                                 economy)
        if llm_only and economy == 0:
            result = _rag_cycles(index, dialog, router, resolution, client, settings, trace, deadline, result)
        return result
    except Exception as exc:  # ядро не роняет прогон (C2)
        trace.event("error", error=f"{type(exc).__name__}: {str(exc)[:200]}")
        best = fallback_node(retrieval, index) if retrieval else None
        text = extractive_fallback(None, index, best) if best else TECHNICAL_ANSWER
        return Result(answer=text, decision="answer", fallback=True, trace=trace,
                      mode="extractive" if best else "technical")
