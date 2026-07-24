"""Typed configuration.

Everything that can change a reported number is declared here as a pydantic
model, so a result is reproducible from (git SHA + config.yaml). Unknown keys
are rejected rather than silently ignored -- a typo'd knob that quietly does
nothing is the easiest way to publish a wrong number.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_CONFIG_PATH = Path("config.yaml")
LOCAL_CONFIG_PATH = Path("config.local.yaml")
ENV_PREFIX = "BUGLOC_"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RepoConfig(_Strict):
    name: str
    url: str
    default_branch: str = "main"


class PathsConfig(_Strict):
    cache_dir: Path = Path(".cache")
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")


class MiningConfig(_Strict):
    max_files_per_commit: int = Field(default=10, ge=1)
    exclude_path_globs: list[str] = Field(default_factory=list)
    source_extensions: list[str] = Field(default_factory=lambda: [".py"])
    fix_patterns: list[str] = Field(default_factory=list)
    exclude_message_patterns: list[str] = Field(default_factory=list)
    skip_merge_commits: bool = True
    min_gold_files: int = Field(default=1, ge=1)
    # Drop a gold file if it does not exist at the parent commit (a file the fix
    # *created* can never be retrieved from the buggy state). Without this the
    # dataset has an unreachable accuracy ceiling.
    require_gold_in_parent: bool = True
    # Remove literal gold file paths from the query text. Commit messages
    # occasionally name the file that was changed, which is direct leakage.
    scrub_gold_paths_from_query: bool = True

    @field_validator("fix_patterns", "exclude_message_patterns")
    @classmethod
    def _patterns_must_compile(cls, v: list[str]) -> list[str]:
        for pattern in v:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc
        return v


class SplitConfig(_Strict):
    strategy: Literal["temporal", "random"] = "temporal"
    dev_fraction: float = Field(default=0.7, gt=0.0, lt=1.0)


class RetrievalConfig(_Strict):
    chunk_max_chars: int = Field(default=2000, ge=100)
    chunk_overlap_chars: int = Field(default=200, ge=0)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = Field(default=10, ge=1)
    rrf_k: int = Field(default=60, ge=1)

    @field_validator("chunk_overlap_chars")
    @classmethod
    def _overlap_smaller_than_chunk(cls, v: int, info) -> int:
        max_chars = info.data.get("chunk_max_chars")
        if max_chars is not None and v >= max_chars:
            raise ValueError("chunk_overlap_chars must be smaller than chunk_max_chars")
        return v


class DbConfig(_Strict):
    dsn: str = "postgresql://bugloc:bugloc@localhost:5433/bugloc"


class LoggingConfig(_Strict):
    level: str = "INFO"


class Config(_Strict):
    seed: int = 42
    paths: PathsConfig = Field(default_factory=PathsConfig)
    repos: list[RepoConfig] = Field(default_factory=list)
    mining: MiningConfig = Field(default_factory=MiningConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    db: DbConfig = Field(default_factory=DbConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def repo(self, name: str) -> RepoConfig:
        for r in self.repos:
            if r.name == name:
                return r
        known = ", ".join(r.name for r in self.repos) or "<none configured>"
        raise KeyError(f"unknown repo {name!r}; configured repos: {known}")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`, returning a new dict."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _env_overrides() -> dict:
    """Read BUGLOC_* env vars for the few settings that are machine-specific.

    Deliberately narrow: only secrets/connection details. Everything that
    affects a metric must come from the config file so it is version controlled
    alongside the result.
    """
    overrides: dict = {}
    dsn = os.environ.get(f"{ENV_PREFIX}DB_DSN")
    if dsn:
        overrides["db"] = {"dsn": dsn}
    level = os.environ.get(f"{ENV_PREFIX}LOG_LEVEL")
    if level:
        overrides["logging"] = {"level": level}
    return overrides


def load_config(path: Path | str | None = None) -> Config:
    """Load config.yaml, layer config.local.yaml over it, then env overrides."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"config file not found: {path} (run from the project root, or pass --config)"
        )

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")

    if path == DEFAULT_CONFIG_PATH and LOCAL_CONFIG_PATH.exists():
        local = yaml.safe_load(LOCAL_CONFIG_PATH.read_text()) or {}
        raw = _deep_merge(raw, local)

    raw = _deep_merge(raw, _env_overrides())
    return Config.model_validate(raw)
