"""Tests for configuration loading.

The config is the reproducibility contract: a published number is only
defensible if the settings behind it were loaded exactly as written. These
tests cover the failure modes that would silently corrupt that -- a typo'd key
being ignored, an override not landing, an impossible chunk setting passing.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from buglocalizer import __version__
from buglocalizer.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_version_is_set():
    assert __version__


def test_defaults_are_sane():
    cfg = Config()
    assert cfg.seed == 42
    assert cfg.split.strategy == "temporal"
    assert 0 < cfg.split.dev_fraction < 1
    assert cfg.mining.max_files_per_commit >= 1


def test_repo_lookup_by_name():
    cfg = Config.model_validate(
        {"repos": [{"name": "flask", "url": "https://example.com/flask.git"}]}
    )
    assert cfg.repo("flask").url.endswith("flask.git")
    with pytest.raises(KeyError):
        cfg.repo("nope")


def test_unknown_key_is_rejected():
    # A silently-ignored typo in config.yaml is how you publish a number that
    # was produced by settings other than the ones you documented.
    with pytest.raises(ValidationError):
        Config.model_validate({"mining": {"max_files_per_comit": 10}})


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValidationError):
        Config.model_validate({"retrieval": {"chunk_max_chars": 500, "chunk_overlap_chars": 500}})


def test_shipped_config_yaml_is_valid():
    """The config.yaml committed to the repo must actually parse."""
    cfg = load_config(REPO_ROOT / "config.yaml")
    assert {r.name for r in cfg.repos} == {"flask", "requests", "pandas"}
    assert cfg.mining.fix_patterns, "fix patterns are what make mining possible"


def test_local_and_env_overrides(tmp_path, monkeypatch):
    base = tmp_path / "config.yaml"
    base.write_text(
        textwrap.dedent("""
        seed: 1
        db:
          dsn: postgresql://from-file/db
        retrieval:
          top_k: 5
        """)
    )
    # Env overrides win over the file, and only touch the key they name.
    monkeypatch.setenv("BUGLOC_DB_DSN", "postgresql://from-env/db")
    cfg = load_config(base)
    assert cfg.db.dsn == "postgresql://from-env/db"
    assert cfg.retrieval.top_k == 5
    assert cfg.seed == 1


def test_missing_config_file_raises_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does-not-exist.yaml")
