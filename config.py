"""Единый источник параметров решения (03-plan §4.5, P9.6).

Все значения читаются из переменных окружения платформы (REG-9).
Локально дополнительно подхватывается файл .env (если установлен python-dotenv).
Ключи никогда не логируются (B5, P5.10).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_dotenv() -> None:
    """Локальная разработка: читаем .env рядом с кодом, не перетирая заданные переменные."""
    path = ROOT / ".env"
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(path, override=False)
    except ImportError:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


def env(name: str, default: str = "") -> str:
    """Значение переменной; пустое или равное собственному имени (неподставленный
    плейсхолдер платформы, ST-1) считается отсутствующим."""
    value = os.getenv(name, "").strip()
    if not value or value == name or value == "${" + name + "}":
        return default
    return value


def first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = env(name)
        if value:
            return value
    return default


def env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(float(env(name)))))
    except ValueError:
        return default


def env_float(name: str, default: float, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(env(name))))
    except ValueError:
        return default


def env_bool(name: str, default: bool = False) -> bool:
    value = env(name).lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on", "да"}


# Минимальный лимит токенов генерации: DeepSeek V4.1 Flash тратит reasoning-токены
# из того же бюджета (03-plan §2, Q7, Q8).
MIN_MAX_TOKENS = 2048


@dataclass(frozen=True)
class Settings:
    # Доступ к API (REG-9, ST-1)
    base_url: str = field(default_factory=lambda: first_env(
        "LLM_BASE_URL", "OPENAI_BASE_URL", "BASE_URL").rstrip("/"))
    llm_key: str = field(default_factory=lambda: first_env("LLM_API_KEY", "OPENAI_API_KEY"))
    emb_key: str = field(default_factory=lambda: first_env(
        "EMBEDDER_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY"))
    rerank_key: str = field(default_factory=lambda: first_env(
        "RERANKER_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY"))

    # Модели организаторов (B1, U-1)
    llm_model: str = field(default_factory=lambda: first_env(
        "LLM_MODEL", "OPENAI_MODEL", "GEN_MODEL", default="deepseek/deepseek-v4.1-flash"))
    emb_model: str = field(default_factory=lambda: env("EMB_MODEL", "baai/bge-m3"))
    rerank_model: str = field(default_factory=lambda: env("RERANKER_MODEL", "qwen/qwen3-reranker-8b"))

    # Генерация (Q7): температура из окружения, лимит не ниже безопасного минимума
    temperature: float = field(default_factory=lambda: env_float("TEMPERATURE", 0.0, 0.0, 2.0))
    max_tokens: int = field(default_factory=lambda: max(
        MIN_MAX_TOKENS, env_int("MAX_TOKENS", 4096, 128, 32768)))

    # Параллелизм (Q6): из окружения, с собственным потолком
    max_workers: int = field(default_factory=lambda: env_int("MAX_WORKERS", 8, 1, 16))

    # Данные
    data_dir: Path = field(default_factory=lambda: Path(env("DATA_DIR", "data")))

    # Надёжность (NFR-3, P9.2, P9.4)
    # http_timeout_s — верхняя граница таймаута одной попытки; фактический — по типу вызова (R2, 03-plan §4.5)
    http_timeout_s: float = field(default_factory=lambda: env_float("HTTP_TIMEOUT_S", 120.0, 5.0, 600.0))
    max_retries: int = field(default_factory=lambda: env_int("MAX_RETRIES", 2, 0, 8))
    chat_timeout_s: float = field(default_factory=lambda: env_float("CHAT_TIMEOUT_S", 60.0, 5.0, 600.0))
    aux_timeout_s: float = field(default_factory=lambda: env_float("AUX_TIMEOUT_S", 20.0, 2.0, 600.0))
    call_budget_chat_s: float = field(default_factory=lambda: env_float("CALL_BUDGET_CHAT_S", 90.0, 5.0, 900.0))
    call_budget_aux_s: float = field(default_factory=lambda: env_float("CALL_BUDGET_AUX_S", 30.0, 2.0, 900.0))
    question_budget_s: float = field(default_factory=lambda: env_float("QUESTION_BUDGET_S", 150.0, 10.0, 1200.0))
    backoff_cap_s: float = field(default_factory=lambda: env_float("BACKOFF_CAP_S", 8.0, 0.5, 60.0))
    # Пачка эмбеддингов индекса (до 64 текстов) идёт дольше одиночного запроса: свой таймаут и бюджет
    embed_index_timeout_s: float = field(default_factory=lambda: env_float("EMBED_INDEX_TIMEOUT_S", 60.0, 2.0, 600.0))
    embed_index_budget_s: float = field(default_factory=lambda: env_float("EMBED_INDEX_BUDGET_S", 120.0, 2.0, 900.0))
    run_deadline_s: float = field(default_factory=lambda: env_float("RUN_DEADLINE_S", 1140, 60, 19.7 * 60))
    # Адаптивный бюджет прогона (R3, 03-plan §4.5)
    platform_limit_s: float = field(default_factory=lambda: env_float("PLATFORM_LIMIT_S", 1200.0, 60.0, 7200.0))
    target_finish_share: float = field(default_factory=lambda: env_float("TARGET_FINISH_SHARE", 0.97, 0.1, 1.0))
    index_share: float = field(default_factory=lambda: env_float("INDEX_SHARE", 0.15, 0.01, 0.9))
    workers_min: int = field(default_factory=lambda: env_int("WORKERS_MIN", 1, 1, 16))

    # Разбиение на узлы (03a §2, 06-implementation-guide §7)
    child_min_chars: int = field(default_factory=lambda: env_int("CHILD_MIN_CHARS", 200, 20, 2000))
    child_max_chars: int = field(default_factory=lambda: env_int("CHILD_MAX_CHARS", 1200, 300, 8000))
    split_overlap: float = field(default_factory=lambda: env_float("SPLIT_OVERLAP", 0.12, 0.0, 0.5))
    # Контекст: адаптивный родитель, малые документы целиком, опись базы (03a §2.2, §3.3)
    parent_max_chars: int = field(default_factory=lambda: env_int("PARENT_MAX_CHARS", 3500, 500, 20000))
    whole_doc_chars: int = field(default_factory=lambda: env_int("WHOLE_DOC_CHARS", 8000, 0, 50000))
    inventory_max_chars: int = field(default_factory=lambda: env_int("INVENTORY_MAX_CHARS", 3000, 300, 20000))
    # Поиск (T1.12; 06 §7): каналы, взвешенный RRF, бусты метаданных, целые документы
    fts_k: int = field(default_factory=lambda: env_int("FTS_K", 40, 5, 200))
    dense_k: int = field(default_factory=lambda: env_int("DENSE_K", 40, 5, 200))
    rrf_k: int = field(default_factory=lambda: env_int("RRF_K", 60, 1, 500))
    rrf_w_dense: float = field(default_factory=lambda: env_float("RRF_W_DENSE", 1.15, 0.0, 10.0))
    rrf_w_fts: float = field(default_factory=lambda: env_float("RRF_W_FTS", 1.0, 0.0, 10.0))
    boost_target: float = field(default_factory=lambda: env_float("BOOST_TARGET", 1.3, 0.1, 5.0))
    penalty_other: float = field(default_factory=lambda: env_float("PENALTY_OTHER", 0.7, 0.0, 1.0))
    max_whole_docs: int = field(default_factory=lambda: env_int("MAX_WHOLE_DOCS", 4, 0, 20))
    # Реранк и бюджет контекста (T1.13; 03b §3)
    rerank_candidates: int = field(default_factory=lambda: env_int("RERANK_CANDIDATES", 40, 5, 200))
    use_rerank_pool_48: bool = field(default_factory=lambda: env_bool("USE_RERANK_POOL_48", False))
    use_hyde: bool = field(default_factory=lambda: env_bool("USE_HYDE", False))
    use_chain_of_note: bool = field(default_factory=lambda: env_bool("USE_CHAIN_OF_NOTE", False))
    use_bge_query_prefix: bool = field(default_factory=lambda: env_bool("USE_BGE_QUERY_PREFIX", False))
    use_best_of_two: bool = field(default_factory=lambda: env_bool("USE_BEST_OF_TWO", False))
    best_of_n: int = field(default_factory=lambda: env_int("BEST_OF_N", 2, 0, 8))  # T3.27: число доп. генераций
    use_stepback: bool = field(default_factory=lambda: env_bool("USE_STEPBACK", False))
    use_draft_judge: bool = field(default_factory=lambda: env_bool("USE_DRAFT_JUDGE", False))
    use_draft_judge_full: bool = field(default_factory=lambda: env_bool("USE_DRAFT_JUDGE_FULL", False))
    use_minimal: bool = field(default_factory=lambda: env_bool("USE_MINIMAL", True))
    use_contextual: bool = field(default_factory=lambda: env_bool("USE_CONTEXTUAL", False))
    use_d2q: bool = field(default_factory=lambda: env_bool("USE_D2Q", False))
    use_prf: bool = field(default_factory=lambda: env_bool("USE_PRF", False))
    use_lean_context: bool = field(default_factory=lambda: env_bool("USE_LEAN_CONTEXT", False))
    use_query2doc: bool = field(default_factory=lambda: env_bool("USE_Q2D", False))
    rerank_top_n: int = field(default_factory=lambda: env_int("RERANK_TOP_N_SLICE", 8, 1, 50))
    rerank_doc_chars: int = field(default_factory=lambda: env_int("RERANK_DOC_CHARS", 1500, 200, 8000))
    context_budget_chars: int = field(default_factory=lambda: env_int("CONTEXT_BUDGET_CHARS", 16000, 2000, 100000))
    # Кандидат PC: квота контекстного родителя при выборе top-N (03b §3); включается только для абляции
    use_rerank_parent_cap: bool = field(default_factory=lambda: env_bool("USE_RERANK_PARENT_CAP", False))
    rerank_parent_cap: int = field(default_factory=lambda: env_int("RERANK_PARENT_CAP", 2, 1, 50))

    # Соседние части нормы (03b §3, кандидат N): к найденной части статьи — первая (общая) и смежные части
    context_neighbors: bool = field(default_factory=lambda: env_bool("CONTEXT_NEIGHBORS", False))
    neighbors_max: int = field(default_factory=lambda: env_int("NEIGHBORS_MAX", 0, 0, 50))  # 0 — без потолка
    # Строгие цели (03b §3, кандидат T): цель распознана однозначно — чужие документы компании не идут в контекст
    strict_targets: bool = field(default_factory=lambda: env_bool("STRICT_TARGETS", False))
    demote_other: bool = field(default_factory=lambda: env_bool("DEMOTE_OTHER", False))  # кандидат D (S7)
    # Инструкция реранкеру (03b §3, кандидат R): Qwen3-Reranker обучен с префиксом «Instruct: … Query: …»
    rerank_instruct: bool = field(default_factory=lambda: env_bool("RERANK_INSTRUCT", False))
    # Подвопросы (03b §3, кандидат S): роутер выдаёт до 3 аспектов; каждый ищется и реранжируется с квотой
    use_subqueries: bool = field(default_factory=lambda: env_bool("USE_SUBQUERIES", False))
    subquery_quota: int = field(default_factory=lambda: env_int("SUBQUERY_QUOTA", 4, 1, 20))
    # Проверки ответа и повторная генерация (T1.15; 03b §5)
    quote_min_chars: int = field(default_factory=lambda: env_int("QUOTE_MIN_CHARS", 20, 1, 500))
    quote_max_chars: int = field(default_factory=lambda: env_int("QUOTE_MAX_CHARS", 200, 20, 2000))
    quote_fuzzy_threshold: int = field(default_factory=lambda: env_int("QUOTE_FUZZY_THRESHOLD", 95, 50, 100))
    max_regenerations: int = field(default_factory=lambda: env_int("MAX_REGENERATIONS", 2, 0, 3))
    # Эмбеддинги узлов (T1.8): размер батча и число параллельных батчей
    embed_batch: int = field(default_factory=lambda: env_int("EMBED_BATCH", 64, 1, 512))
    embed_workers: int = field(default_factory=lambda: env_int("EMBED_WORKERS", 4, 1, 16))

    # Условность и границы отрицательных выводов; OFF до приёмки.
    use_conditional_example_v3: bool = field(default_factory=lambda: env_bool("USE_CONDITIONAL_EXAMPLE_V3", False))

    # Дополнительный пример отсутствующего документа; OFF до приёмки.
    use_conditional_example_v2: bool = field(default_factory=lambda: env_bool("USE_CONDITIONAL_EXAMPLE_V2", False))

    # Синтетический пример условного ответа; OFF до приёмки.
    use_conditional_example: bool = field(default_factory=lambda: env_bool("USE_CONDITIONAL_EXAMPLE", False))

    # Граница исторической редакции: исходный эксперимент сохранён, интеграционный default OFF.
    use_historical_law_limit: bool = field(default_factory=lambda: env_bool("USE_HISTORICAL_LAW_LIMIT", False))

    # Конкретные недостающие данные для ответа; OFF до приёмки.
    use_actionable_gaps: bool = field(default_factory=lambda: env_bool("USE_ACTIONABLE_GAPS", False))

    # Выборочная проверка условий санкций; OFF до приёмки.
    use_sanction_condition_review: bool = field(default_factory=lambda: env_bool("USE_SANCTION_CONDITION_REVIEW", False))

    # Кандидат: условия применимости правовой нормы, OFF до проверки.
    use_applicability_guard: bool = field(default_factory=lambda: env_bool("USE_APPLICABILITY_GUARD", False))

    # Кандидат: условный прямой ответ при неизвестных обстоятельствах, OFF до проверки.
    use_conditional_first: bool = field(default_factory=lambda: env_bool("USE_CONDITIONAL_FIRST", False))

    # Кандидат D: разнообразие документов после реранка, только для абляции.
    use_rerank_doc_diversity: bool = field(default_factory=lambda: env_bool("USE_RERANK_DOC_DIVERSITY", False))

    # D2: разнообразие положений в уже представленных документах, только для абляции.
    use_rerank_parent_diversity: bool = field(default_factory=lambda: env_bool("USE_RERANK_PARENT_DIVERSITY", False))

    # C: инструкции по условиям и раздельной опоре, для абляции.
    use_claim_discipline: bool = field(default_factory=lambda: env_bool("USE_CLAIM_DISCIPLINE", False))


    # SAFE_CONTEXT: ограниченное выделение метаинструкций, кандидат OFF.
    use_safe_context: bool = field(default_factory=lambda: env_bool("USE_SAFE_CONTEXT", False))

    # Кандидат M: границы свободного поля missing, включается только для абляции.
    use_scoped_missing: bool = field(default_factory=lambda: env_bool("USE_SCOPED_MISSING", False))
    use_norm_analysis: bool = field(default_factory=lambda: env_bool("USE_NORM_ANALYSIS", True))
    # T3.20: служебное поле facts в JSON генерации, кандидат OFF (контракт — 05-tasks T3.20).
    use_typed_facts: bool = field(default_factory=lambda: env_bool("USE_TYPED_FACTS", False))
    # Q19: ответ по трактовкам вместо уточнения (вебинар 11.11: уточнение = отсутствие ответа). Включён по умолчанию
    # решением пользователя 11.11.2026; USE_ANSWER_VARIANTS=0 возвращает прежний путь для сравнения.
    use_answer_variants: bool = field(default_factory=lambda: env_bool("USE_ANSWER_VARIANTS", True))
    # Эксперимент «всё на DeepSeek»: расширение запроса, выбор по карте базы и ранжирование силами LLM (без реранкера
    # и эмбеддера), большой контекст, повторные циклы RAG. Выключен; в сабмите-эксперименте включается умолчанием.
    use_llm_only: bool = field(default_factory=lambda: env_bool("USE_LLM_ONLY", False))
    llm_rag_cycles: int = field(default_factory=lambda: env_int("LLM_RAG_CYCLES", 2, 0, 4))
    # Проверяющий проход: вторая модель сверяет ответ с положениями; при замечаниях — перегенерация (critic.py).
    use_critic: bool = field(default_factory=lambda: env_bool("USE_CRITIC", False))
    critic_rounds: int = field(default_factory=lambda: env_int("CRITIC_ROUNDS", 2, 1, 4))
    # Полнота ответа: правило 10 допускает до 6 утверждений и до ~200 слов. Платформа 11.11.2026: 37,474% за 569,6 с
    # против 38,434% за 447,3 с у того же Q19 без неё: выигрыша нет, прогон медленнее. Выключен по умолчанию.
    use_complete_answer: bool = field(default_factory=lambda: env_bool("USE_COMPLETE_ANSWER", False))

    # Только для локальной разработки: дисковый кэш ответов API (P9.5)
    dev_cache: bool = field(default_factory=lambda: env_bool("DEV_CACHE", False))
    dev_cache_dir: Path = field(default_factory=lambda: Path(env("DEV_CACHE_DIR", str(ROOT / "dev_cache"))))

    def describe(self) -> dict:
        """Параметры для журнала прогона — без ключей."""
        return {
            "base_url_set": bool(self.base_url),
            "llm_key_set": bool(self.llm_key),
            "llm_model": self.llm_model,
            "emb_model": self.emb_model,
            "rerank_model": self.rerank_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_workers": self.max_workers,
            "data_dir": str(self.data_dir),
            "run_deadline_s": self.run_deadline_s,
            "platform_limit_s": self.platform_limit_s, "target_finish_share": self.target_finish_share,
            "index_share": self.index_share, "workers_min": self.workers_min,
            "http_timeout_s": self.http_timeout_s, "max_retries": self.max_retries,
            "chat_timeout_s": self.chat_timeout_s, "aux_timeout_s": self.aux_timeout_s,
            "call_budget_chat_s": self.call_budget_chat_s, "call_budget_aux_s": self.call_budget_aux_s,
            "question_budget_s": self.question_budget_s, "backoff_cap_s": self.backoff_cap_s,
            "embed_index_timeout_s": self.embed_index_timeout_s, "embed_index_budget_s": self.embed_index_budget_s,
            "child_min_chars": self.child_min_chars,
            "child_max_chars": self.child_max_chars,
            "split_overlap": self.split_overlap,
            "parent_max_chars": self.parent_max_chars,
            "whole_doc_chars": self.whole_doc_chars,
            "inventory_max_chars": self.inventory_max_chars,
            "fts_k": self.fts_k, "dense_k": self.dense_k, "rrf_k": self.rrf_k,
            "rrf_w_dense": self.rrf_w_dense, "rrf_w_fts": self.rrf_w_fts,
            "boost_target": self.boost_target, "penalty_other": self.penalty_other,
            "max_whole_docs": self.max_whole_docs,
            "rerank_candidates": self.rerank_candidates, "rerank_top_n": self.rerank_top_n,
            "use_rerank_pool_48": self.use_rerank_pool_48,
            "use_hyde": self.use_hyde,
            "use_chain_of_note": self.use_chain_of_note,
            "use_bge_query_prefix": self.use_bge_query_prefix,
            "use_best_of_two": self.use_best_of_two,
            "best_of_n": self.best_of_n,
            "use_stepback": self.use_stepback,
            "use_draft_judge": self.use_draft_judge,
            "use_draft_judge_full": self.use_draft_judge_full,
            "use_minimal": self.use_minimal,
            "use_contextual": self.use_contextual,
            "use_d2q": self.use_d2q,
            "use_prf": self.use_prf,
            "use_lean_context": self.use_lean_context,
            "use_query2doc": self.use_query2doc,
            "rerank_doc_chars": self.rerank_doc_chars, "context_budget_chars": self.context_budget_chars,
            "use_rerank_parent_cap": self.use_rerank_parent_cap,
            "rerank_parent_cap": self.rerank_parent_cap,

            "context_neighbors": self.context_neighbors, "neighbors_max": self.neighbors_max, "strict_targets": self.strict_targets, "demote_other": self.demote_other,
            "rerank_instruct": self.rerank_instruct, "use_subqueries": self.use_subqueries,
            "subquery_quota": self.subquery_quota,
            "quote_min_chars": self.quote_min_chars, "quote_fuzzy_threshold": self.quote_fuzzy_threshold,
            "max_regenerations": self.max_regenerations,
            "embed_batch": self.embed_batch,
            "embed_workers": self.embed_workers,
            "dev_cache": self.dev_cache,
            "use_scoped_missing": self.use_scoped_missing,
            "use_norm_analysis": self.use_norm_analysis,
            "use_typed_facts": self.use_typed_facts,
            "use_safe_context": self.use_safe_context,
            "use_conditional_first": self.use_conditional_first,
            "use_applicability_guard": self.use_applicability_guard,
            "use_sanction_condition_review": self.use_sanction_condition_review,
            "use_actionable_gaps": self.use_actionable_gaps,
            "use_historical_law_limit": self.use_historical_law_limit,
            "use_conditional_example": self.use_conditional_example,
            "use_conditional_example_v2": self.use_conditional_example_v2,
            "use_conditional_example_v3": self.use_conditional_example_v3,
            "use_answer_variants": self.use_answer_variants,
            "use_complete_answer": self.use_complete_answer,

            "use_claim_discipline": self.use_claim_discipline,
            "use_rerank_doc_diversity": self.use_rerank_doc_diversity,
            "use_rerank_parent_diversity": self.use_rerank_parent_diversity,

            "use_llm_only": self.use_llm_only, "llm_rag_cycles": self.llm_rag_cycles,
            "use_critic": self.use_critic, "critic_rounds": self.critic_rounds,
        }


def get_settings() -> Settings:
    return Settings()
