"""Persisting a profile version to disk -- the archive side of `jobsearch profile`.

profile.py is the domain model, extractor.py turns CV text into one via an
LLM; this is the third concern: copying the source files and the extracted
profile.json into a permanent, timestamped home under
config.profile_storage_path, so overwriting profile.json to switch CVs never
destroys anything. `use-profile` can always copy an old version back.
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .profile import CandidateProfile

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(name: str) -> str:
    slug = _SLUG_RE.sub("_", name.lower()).strip("_")
    return slug or "profile"


def archive_profile(storage_root: Path, profile: CandidateProfile) -> tuple[str, Path]:
    """Copy profile.source_files and the extracted profile itself into a new
    directory under storage_root, named `<UTC timestamp>-<slug of first
    source filename>`.

    Returns (profile_id, dest_dir); the caller (cli.main.profile) passes
    profile_id, profile.hash(), and the copied filenames to
    Store.save_profile_record so the archive becomes listable.

    Collision-proof against two archives in the same second: if the
    generated directory already exists, a numeric suffix is appended
    (-2, -3, ...) rather than overwritten -- an archive must never destroy
    a previous one, that would defeat the entire feature.
    """
    storage_root = Path(storage_root)
    storage_root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = _slugify(Path(profile.source_files[0]).stem) if profile.source_files else "profile"
    base_id = f"{stamp}-{slug}"

    profile_id, dest_dir, n = base_id, storage_root / base_id, 2
    while dest_dir.exists():
        profile_id = f"{base_id}-{n}"
        dest_dir = storage_root / profile_id
        n += 1

    dest_dir.mkdir(parents=True)
    for src in profile.source_files:
        src = Path(src)
        shutil.copy2(src, dest_dir / src.name)
    profile.save(dest_dir / "profile.json")

    return profile_id, dest_dir
