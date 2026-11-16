"""Сборка и самопроверка ZIP сабмита (T1.17; REG-7, B5). Запуск из командной строки, в сабмит не входит.

python build_submission.py                 — собрать .local/submission-stage/ и build/submission_<коммит>.zip, проверить
python build_submission.py --no-vectors    — без готовых векторов (имя с суффиксом -novec)
python build_submission.py --check FILE    — только проверить готовый ZIP

Состав ZIP (всё в корне, без папок):
- корневые .py, которые solution.py импортирует прямо или косвенно (граф импортов, не маска *.py:
  инструменты разработки и этот скрипт в архив не попадают);
- requirements.txt — только строки корневого файла, чьи пакеты импортируют модули архива (R5; строки дословно, B4);
- готовые векторы и индексы не включаются: строятся по входной базе при запуске.
  Исторический --no-vectors совместим; --embed отклоняется.

Самопроверка: состав и плоская структура, размер ≤ 128 МБ, нет базы, вопросов, ответов и .env, нет значений
секретов из .env ни в одном файле (побайтово), все импорты — stdlib, модули архива или пакеты requirements.txt,
нет запрещённых в ядре библиотек, все модули компилируются.
"""

from __future__ import annotations

import argparse
import ast
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
STAGE = ROOT / ".local" / "submission-stage"
# Имя архива — по коммиту (правило COLLAB.md §4 п.8): submission_<коммит>[-novec][-dirty].zip.
# По имени на лидерборде видно, какой код отправлен; -dirty — модули архива отличаются от коммита, не отправлять.
ZIP_PREFIX = "submission"
ENTRY = "solution.py"
VECTORS = "emb_cache.npz"
MAX_ZIP_BYTES = 128 * 1024 * 1024
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
FORBIDDEN_IMPORTS = {"tiktoken", "chromadb", "langchain", "langchain_openai", "langchain_text_splitters", "faiss"}
FORBIDDEN_NAMES = {".env", "questions.csv", "demo_questions.csv", "demo_output.csv", "submission.csv"}
FORBIDDEN_SUFFIXES = {".zip", ".docx", ".md", ".csv", ".pdf", ".env"}
# имя пакета в requirements.txt → имя модуля при импорте
IMPORT_NAMES = {"python-docx": "docx", "beautifulsoup4": "bs4", "PyYAML": "yaml", "scikit-learn": "sklearn",
                "python-dotenv": "dotenv", "python-multipart": "multipart", "rank-bm25": "rank_bm25",
                "faiss-cpu": "faiss", "SQLAlchemy": "sqlalchemy"}


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------- состав ----------

def imported_modules(source: str) -> set[str]:
    """Имена верхнего уровня всех импортов модуля (включая импорты внутри функций и try)."""
    out: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            out.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module.split(".")[0])
    return out


def runtime_modules(root: Path = ROOT, entry: str = ENTRY) -> list[str]:
    """Корневые .py, достижимые из solution.py по графу импортов."""
    local = {p.stem for p in root.glob("*.py")}
    seen: set[str] = set()
    stack = [Path(entry).stem]
    while stack:
        name = stack.pop()
        if name in seen or name not in local:
            continue
        seen.add(name)
        stack.extend(imported_modules((root / f"{name}.py").read_text(encoding="utf-8")) & local)
    return sorted(f"{n}.py" for n in seen)


def requirement_lines(text: str) -> dict[str, str]:
    """Имя модуля при импорте → строка requirements.txt как есть."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        pkg = line.split("==")[0].split(">=")[0].split("[")[0].strip()
        out[IMPORT_NAMES.get(pkg, pkg.replace("-", "_").lower())] = raw.strip()
    return out


def requirement_modules(path: Path) -> set[str]:
    return set(requirement_lines(path.read_text(encoding="utf-8")))


def third_party_imports(sources: list[str], local: set[str]) -> set[str]:
    mods: set[str] = set()
    for source in sources:
        mods |= {m for m in imported_modules(source) if m not in local and m not in sys.stdlib_module_names}
    return mods


def minimal_requirements(modules: list[str]) -> str:
    """R5: строки корневого requirements.txt для пакетов, которые импортируют модули архива (в порядке файла)."""
    lines = requirement_lines((ROOT / "requirements.txt").read_text(encoding="utf-8"))
    local = {Path(m).stem for m in modules}
    needed = third_party_imports([(ROOT / m).read_text(encoding="utf-8") for m in modules], local)
    return "".join(line + "\n" for mod, line in lines.items() if mod in needed)


# ---------- готовые векторы ----------

def ship_vectors(dest: Path, embed: bool) -> str:
    """Векторы узлов выданной базы → dest. Возвращает строку для отчёта."""
    sys.path.insert(0, str(ROOT))
    from config import get_settings
    from emb_cache import EmbeddingCache, text_key
    from indexer import build_index
    from kb_loader import load_kb

    s = get_settings()
    index = build_index(load_kb(ROOT / "knowledge-base.zip"), None, s)
    texts = [index.search_text(n.node_id) for n in index.searchable_nodes()]
    cache = EmbeddingCache(s.emb_model, paths=[ROOT / "dev_cache" / VECTORS])
    missing = [t for t in texts if cache.get(t) is None]
    if missing and embed:
        from llm_client import LLMClient
        client = LLMClient(s)
        for i in range(0, len(missing), s.embed_batch):
            chunk = missing[i:i + s.embed_batch]
            for t, v in zip(chunk, client.embed(chunk, batch_size=len(chunk))):
                cache.put(t, v)
        missing = [t for t in texts if cache.get(t) is None]
    keys = {text_key(t) for t in texts if cache.get(t) is not None}
    if not keys or not cache.save(dest, only_keys=keys):
        return f"готовых векторов нет (узлов {len(texts)}) — индекс строится при запуске"
    return f"готовые векторы: {len(keys)}/{len(texts)} узлов, модель {s.emb_model}" + \
        (f"; без вектора {len(missing)} (досчитать: --embed)" if missing else "")


# ---------- самопроверка ----------

def env_secrets() -> list[bytes]:
    path = ROOT / ".env"
    out = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                v = v.strip().strip("'\"")
                if any(m in k.upper() for m in SECRET_MARKERS) and len(v) >= 8:
                    out.append(v.encode("utf-8"))
    return out


def check_zip(zip_path: Path) -> list[str]:
    """Пустой список — ZIP годен. Секреты в сообщениях не выводятся."""
    problems: list[str] = []
    size = zip_path.stat().st_size
    if size > MAX_ZIP_BYTES:
        problems.append(f"размер {size / 2**20:.1f} МБ > 128 МБ")
    secrets = env_secrets()
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        files = {n: zf.read(n) for n in names}
    if ENTRY not in names:
        problems.append(f"нет {ENTRY}")
    if "requirements.txt" not in names:
        problems.append("нет requirements.txt")
    else:
        root_lines = {line.strip() for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()}
        for line in files["requirements.txt"].decode("utf-8").splitlines():
            if line.strip() and line.strip() not in root_lines:
                problems.append(f"requirements.txt: строки «{line.strip()}» нет в корневом файле")
    for n in names:
        if "/" in n or "\\" in n:
            problems.append(f"вложенный путь: {n}")
        if n in FORBIDDEN_NAMES or Path(n).suffix.lower() in FORBIDDEN_SUFFIXES or n.startswith("."):
            problems.append(f"недопустимый файл: {n}")
        if any(sec in files[n] for sec in secrets):
            problems.append(f"в {n} найдено значение секрета из .env")
        if b"sk-or-" in files[n]:
            problems.append(f"в {n} похожая на ключ OpenRouter строка")

    allowed = set(requirement_lines(files.get("requirements.txt", b"").decode("utf-8")))
    local = {Path(n).stem for n in names if n.endswith(".py")}
    for n in sorted(n for n in names if n.endswith(".py")):
        source = files[n].decode("utf-8")
        try:
            compile(source, n, "exec")
        except SyntaxError as exc:
            problems.append(f"{n}: не компилируется ({exc.msg}, строка {exc.lineno})")
            continue
        for mod in sorted(imported_modules(source)):
            if mod in FORBIDDEN_IMPORTS:
                problems.append(f"{n}: запрещённый в ядре импорт {mod}")
            elif mod not in local and mod not in allowed and mod not in sys.stdlib_module_names:
                problems.append(f"{n}: импорт {mod} не из stdlib, архива или requirements.txt архива")
    missing = set(runtime_modules_from(files)) - set(names)
    if missing:
        problems.append(f"не хватает модулей: {sorted(missing)}")
    return problems


def runtime_modules_from(files: dict[str, bytes]) -> list[str]:
    """Модули, которые импортируют .py архива и которые есть в корне проекта, но не в архиве."""
    project = {p.stem for p in ROOT.glob("*.py")}
    need = set()
    for n, data in files.items():
        if n.endswith(".py"):
            try:
                need |= imported_modules(data.decode("utf-8")) & project
            except (SyntaxError, UnicodeDecodeError):
                continue  # уже отмечено как ошибка компиляции
    return sorted(f"{m}.py" for m in need)


# ---------- сборка ----------

def git_commit_tag(files: list[str]) -> tuple[str, bool]:
    """Короткий хэш HEAD и признак незакоммиченных изменений в файлах архива."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--", *files], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit", True
    return commit or "nogit", bool(status)


def zip_path_for(commit: str, dirty: bool, vectors: bool) -> Path:
    name = f"{ZIP_PREFIX}_{commit}" + ("" if vectors else "-novec") + ("-dirty" if dirty else "")
    return BUILD / f"{name}.zip"


def ship_precomputed(stage: Path, settings) -> str:
    """Only active, source-bound enrichment assets belong in an experimental archive."""
    import json
    import numpy as np
    notes = []
    if getattr(settings, "use_contextual", False):
        source = ROOT / "contextual_cache.json"
        data = json.loads(source.read_text(encoding="utf-8"))
        if data.get("schema_version") != 2 or not isinstance(data.get("contexts"), dict):
            raise ValueError("Contextual candidate requires source-bound v2 data")
        shutil.copy2(source, stage / source.name)
        notes.append("contextual v2")
    if getattr(settings, "use_d2q", False):
        source = ROOT / "d2q_vectors.npz"
        with np.load(source, allow_pickle=False) as data:
            if ("version" not in data or int(data["version"].item()) != 2
                    or not {"node_ids", "node_text_hashes", "model", "vectors"}.issubset(data.files)):
                raise ValueError("D2Q candidate requires source-bound v2 vectors")
        shutil.copy2(source, stage / source.name)
        notes.append("d2q v2")
    return "; " + ", ".join(notes) if notes else ""


def build(vectors: bool = False, embed: bool = False) -> Path:
    if vectors or embed:
        raise ValueError("Финальные сабмиты строят индекс на входной базе; предзагрузка отключена")
    stage = STAGE.resolve()
    if not stage.is_relative_to(ROOT.resolve()) or stage == ROOT.resolve():
        raise ValueError("Каталог сборки вне проекта")
    if STAGE.exists():
        shutil.rmtree(stage)
    STAGE.mkdir(parents=True)
    modules = runtime_modules()
    for name in modules:
        shutil.copy2(ROOT / name, STAGE / name)
    (STAGE / "requirements.txt").write_text(minimal_requirements(modules), encoding="utf-8", newline="\n")
    note = ship_vectors(STAGE / VECTORS, embed) if vectors else "готовые векторы не включены (--no-vectors)"
    ctx_note = "; только код: индекс и эмбеддинги строятся при запуске"
    commit, dirty = git_commit_tag(modules + ["requirements.txt"])
    zip_path = zip_path_for(commit, dirty, vectors)
    zip_path.parent.mkdir(exist_ok=True)
    tmp = zip_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(STAGE.iterdir()):
            zf.write(path, arcname=path.name)  # в корень архива, без папок (REG-7)
    tmp.replace(zip_path)
    log(f"модулей: {len(modules)} — {', '.join(modules)}")
    log("пакеты requirements.txt: " + (", ".join((STAGE / "requirements.txt").read_text(encoding="utf-8").split()) or "нет"))
    log(note + ctx_note)
    if dirty:
        log("ВНИМАНИЕ: модули архива отличаются от коммита — архив помечен -dirty, на платформу не отправлять")
    return zip_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-vectors", action="store_true", help="совместимость: финальная сборка всегда без векторов")
    ap.add_argument("--embed", action="store_true", help="устаревший режим, в финальной сборке запрещён")
    ap.add_argument("--check", metavar="ZIP", help="только проверить готовый ZIP")
    args = ap.parse_args()

    zip_path = Path(args.check) if args.check else build(False, args.embed)
    problems = check_zip(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        count = len(zf.namelist())
    size = zip_path.stat().st_size / 2**20
    if problems:
        log(f"{zip_path}: самопроверка НЕ пройдена ({count} файлов, {size:.2f} МБ):")
        for p in problems:
            log(f"  - {p}")
        sys.exit(1)
    log(f"{zip_path}: самопроверка пройдена — {count} файлов в корне, {size:.2f} МБ, секретов нет")
    if not args.check:
        log(f"проверка в эмуляторе: python eval/run_platform.py --solution {STAGE.relative_to(ROOT)} --name <имя>")


if __name__ == "__main__":
    main()
