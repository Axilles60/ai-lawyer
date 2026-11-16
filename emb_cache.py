"""Кэш эмбеддингов по хэшу текста (T1.8, R6; 03a §1, Q4).

Ключ — sha256(текст) без ID модели (R6): на платформе модель может называться иначе (`BAAI/bge-m3` против
`baai/bge-m3`), и это не должно отбрасывать кэш. ID модели хранится в файле как справка.
Источники кэша:
- `emb_cache.npz` рядом с кодом — готовые векторы, положенные в ZIP. Они **непроверенные**: прежде чем ими
  пользоваться, indexer.embed_nodes пересчитывает несколько узлов и сравнивает косинусы (совместимость с
  эмбеддером платформы); не прошли — отбрасываются;
- `dev_cache/emb_cache.npz` — локальный кэш разработки (DEV_CACHE=1), дополняется после сборки; не проверяется.
Векторы хранятся в float16 и L2-нормированы.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SHIPPED_CACHE = ROOT / "emb_cache.npz"
KEY_VERSION = "text-v2"  # ключ без ID модели; файлы прежнего формата (ключ с моделью) не читаются


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def usable_vector(vector) -> bool:
    """A real, finite, nonempty 1-D vector with nonzero norm, without squaring overflow."""
    try:
        value = np.asarray(vector)
        return bool(value.ndim == 1 and value.size
                    and (np.issubdtype(value.dtype, np.floating) or np.issubdtype(value.dtype, np.integer))
                    and np.isfinite(value).all() and np.any(value != 0))
    except (TypeError, ValueError, OverflowError):
        return False


def stored_vector(vector):
    if not usable_vector(vector):
        return None
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        value = np.asarray(vector).astype(np.float16)
    return value if usable_vector(value) else None


class EmbeddingCache:
    def __init__(self, model: str, paths: list[Path] | None = None, write_path: Path | None = None):
        self.model = model
        self.write_path = write_path
        self.vectors: dict[str, np.ndarray] = {}
        self.loaded_from: list[str] = []
        self.unverified: set[str] = set()   # ключи из файлов `paths` (готовые векторы ZIP)
        self.file_models: list[str] = []    # ID моделей из прочитанных файлов — справка для отчёта
        for path in paths if paths is not None else [SHIPPED_CACHE]:
            self._load(path, verified=False)
        if write_path is not None:
            self._load(write_path, verified=True)
        self.new: dict[str, np.ndarray] = {}

    def _load(self, path: Path, verified: bool) -> None:
        if not path or not Path(path).is_file():
            return
        try:
            with np.load(path, allow_pickle=False) as data:
                version = str(data["key_version"]) if "key_version" in data.files else ""
                if version != KEY_VERSION:
                    print(f"[emb_cache] кэш {Path(path).name} в прежнем формате ключей — не используется",
                          file=sys.stderr)
                    return
                keys, mat = data["keys"], data["vectors"]
                for k, v in zip(keys, mat):
                    v = stored_vector(v)
                    if v is None:
                        continue
                    key = str(k)
                    if verified:
                        self.vectors[key] = v
                        self.unverified.discard(key)
                    elif key not in self.vectors:
                        self.vectors[key] = v
                        self.unverified.add(key)
                self.file_models.append(str(data["model"]) if "model" in data.files else "")
            self.loaded_from.append(str(path))
        except Exception as exc:  # кэш — только ускорение: любой сбой чтения не должен ронять сборку (C2)
            print(f"[emb_cache] кэш {path} не прочитан: {type(exc).__name__}", file=sys.stderr)
            return

    def get(self, text: str) -> np.ndarray | None:
        return self.vectors.get(text_key(text))

    def is_unverified(self, text: str) -> bool:
        return text_key(text) in self.unverified

    def verify(self, texts: list[str]) -> None:
        """Готовые векторы этих текстов прошли проверку совместимости."""
        self.unverified.difference_update(text_key(t) for t in texts)

    def drop_unverified(self) -> int:
        """Отбросить все непроверенные векторы (несовместимы или проверить нечем). Возвращает их число."""
        for key in self.unverified:
            self.vectors.pop(key, None)
        dropped = len(self.unverified)
        self.unverified.clear()
        return dropped

    def put(self, text: str, vector: np.ndarray) -> None:
        key = text_key(text)
        vec = stored_vector(vector)
        if vec is None:
            return
        self.vectors[key] = vec
        self.new[key] = vec
        self.unverified.discard(key)

    def discard(self, texts) -> int:
        """Забыть векторы этих текстов (несовместимы с эмбеддером): не попадут в файл при сохранении."""
        dropped = 0
        for text in texts:
            key = text_key(text)
            if key in self.vectors:
                dropped += 1
            self.vectors.pop(key, None)
            self.new.pop(key, None)
            self.unverified.discard(key)
        return dropped

    def save(self, path: Path | None = None, only_keys: set[str] | None = None) -> bool:
        """Атомарная запись всего кэша (или подмножества ключей) — tmp → os.replace.
        Векторы разной размерности (остатки от иной модели: ключ без ID модели, R6) в один файл
        не складываются — пишется доминирующая форма; сбой записи не роняет сборку (C2)."""
        path = Path(path or self.write_path or "")
        if not str(path) or not self.vectors:
            return False
        keys = sorted(k for k in self.vectors if (only_keys is None or k in only_keys)
                      and stored_vector(self.vectors[k]) is not None)
        if not keys:
            return False
        dominant = Counter(self.vectors[k].shape for k in keys).most_common(1)[0][0]
        keys = [k for k in keys if self.vectors[k].shape == dominant]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npz")
            os.close(fd)
            with open(tmp, "wb") as f:
                np.savez(f, model=np.array(self.model), key_version=np.array(KEY_VERSION), keys=np.array(keys),
                         vectors=np.vstack([self.vectors[k] for k in keys]).astype(np.float16))
            os.replace(tmp, path)
            return True
        except (OSError, ValueError):
            return False
