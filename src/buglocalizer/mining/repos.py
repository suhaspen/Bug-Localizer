"""Clone and refresh the target repositories into the cache directory.

Full clones, not shallow or blobless ones: mining needs the whole commit graph,
and Milestone 2 needs to read file *contents* at thousands of historical
commits. A `--filter=blob:none` clone would turn that into a network fetch per
file and break the offline guarantee.
"""

from __future__ import annotations

from pathlib import Path

from git import Repo

from buglocalizer.config import RepoConfig
from buglocalizer.logging_setup import get_logger

log = get_logger(__name__)


def ensure_repo(repo_cfg: RepoConfig, cache_dir: Path, fetch: bool = True) -> Path:
    """Return a local path to the repo, cloning it if absent."""
    path = cache_dir / repo_cfg.name

    if not path.exists():
        log.info("cloning %s → %s (this takes a few minutes for large repos)", repo_cfg.url, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Repo.clone_from(repo_cfg.url, path)
        log.info("cloned %s", repo_cfg.name)
        return path

    if fetch:
        log.info("fetching %s", repo_cfg.name)
        try:
            Repo(path).remotes.origin.fetch()
        except Exception as exc:  # offline is fine — mine what we already have
            log.warning("fetch failed for %s (%s); using the cached clone", repo_cfg.name, exc)

    return path
