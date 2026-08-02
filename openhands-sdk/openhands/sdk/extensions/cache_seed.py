"""Seed the extension cache from object storage before cloning.

An extension is fetched by cloning its repository, so the first fetch on any new
machine pays a clone: a fresh sandbox pays it for every extension, every time,
and a control plane pays it once per extension per deployment.

Where a deployment publishes extension tarballs to object storage, this unpacks
one into the cache first, and the fetch that follows finds the repository already
there. Configured by environment, absent by default, and best effort throughout —
nothing here can turn a working clone into a failure.

Enable by setting ``OH_EXTENSION_CACHE_BUCKET``. Objects are read from
``<prefix>/<cache-dir-name>/latest.tar``, where the directory name is the one
:func:`openhands.sdk.extensions.fetch.get_cache_path` derives from the URL, so a
publisher and a reader agree without exchanging anything.
"""

from __future__ import annotations

import io
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

BUCKET_ENV = "OH_EXTENSION_CACHE_BUCKET"
PREFIX_ENV = "OH_EXTENSION_CACHE_PREFIX"
DEFAULT_PREFIX = "artifacts/plugins"


def seed_cache_from_object_storage(cache_path: Path) -> bool:
    """Unpack a published tarball into ``cache_path``.

    Returns whether the cache now holds a seeded repository. False covers every
    ordinary case — no bucket configured, nothing published for this extension,
    no credentials — and the caller simply clones as it would have.
    """
    bucket = os.getenv(BUCKET_ENV)
    if not bucket:
        return False
    if cache_path.exists():
        # Already a working clone; replacing it would only risk pulling it out
        # from under a concurrent read.
        return False

    key = f"{os.getenv(PREFIX_ENV, DEFAULT_PREFIX)}/{cache_path.name}/latest.tar"
    try:
        import boto3  # pyright: ignore[reportMissingImports]

        response = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        payload = response["Body"].read()
    except Exception as e:
        logger.debug("No seeded extension cache at %s: %s", key, e)
        return False

    try:
        _extract(payload, cache_path)
    except Exception as e:
        logger.warning("Could not unpack the seeded extension cache %s: %s", key, e)
        return False

    logger.info("Seeded extension cache %s from object storage", cache_path.name)
    return True


def _extract(payload: bytes, destination: Path) -> None:
    """Unpack beside the destination, then move it in.

    A reader that finds the directory must find a whole repository: extracting in
    place would let a concurrent fetch mistake a half-written tree for a cache.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r") as archive:
            _extract_safely(archive, staging)
        try:
            staging.rename(destination)
        except OSError:
            # Another process got there first; its copy is as good as ours.
            if not destination.exists():
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _extract_safely(archive: tarfile.TarFile, destination: Path) -> None:
    """Refuse any member that would be written outside ``destination``.

    Object storage is shared infrastructure, so a tarball is untrusted input even
    when this same project wrote it: a member named ``../../.ssh/authorized_keys``
    would otherwise escape the cache directory.
    """
    root = destination.resolve()
    for member in archive.getmembers():
        target = (root / member.name).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"Tar member escapes the cache directory: {member.name}")
        if member.issym() or member.islnk():
            link = (target.parent / member.linkname).resolve()
            if not link.is_relative_to(root):
                raise ValueError(f"Tar link escapes the cache directory: {member.name}")
    # 'data' rejects device nodes and absolute/parent paths on top of the
    # checks above, and is the default from Python 3.14.
    archive.extractall(destination, filter="data")
