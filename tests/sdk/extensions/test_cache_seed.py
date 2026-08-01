"""Seeding the extension cache from object storage.

The seed only ever saves a clone; it must never cause one to fail, so every
unhappy path here asserts a quiet False rather than an exception.
"""

import io
import tarfile
from pathlib import Path

import pytest

from openhands.sdk.extensions.cache_seed import (
    BUCKET_ENV,
    PREFIX_ENV,
    seed_cache_from_object_storage,
)


def build_tarball(entries: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, content in entries.items():
            payload = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


class FakeS3:
    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects = objects or {}
        self.requested: list[str] = []

    def get_object(self, Bucket: str, Key: str):  # noqa: N803 - boto3 signature
        self.requested.append(Key)
        if Key not in self.objects:
            raise KeyError(Key)
        return {"Body": io.BytesIO(self.objects[Key])}


@pytest.fixture
def s3(monkeypatch):
    client = FakeS3()

    class FakeBoto3:
        @staticmethod
        def client(service: str):
            assert service == "s3"
            return client

    monkeypatch.setitem(__import__("sys").modules, "boto3", FakeBoto3)
    return client


def test_without_a_bucket_nothing_happens(tmp_path, monkeypatch):
    monkeypatch.delenv(BUCKET_ENV, raising=False)
    assert seed_cache_from_object_storage(tmp_path / "repo-abc") is False


def test_a_published_tarball_becomes_the_cache(tmp_path, monkeypatch, s3):
    monkeypatch.setenv(BUCKET_ENV, "bucket")
    monkeypatch.delenv(PREFIX_ENV, raising=False)
    s3.objects["artifacts/plugins/repo-abc/latest.tar"] = build_tarball(
        {"README.md": "hello", ".plugin/marketplace.json": "{}"}
    )

    destination = tmp_path / "repo-abc"
    assert seed_cache_from_object_storage(destination) is True
    assert (destination / "README.md").read_text() == "hello"
    assert (destination / ".plugin" / "marketplace.json").read_text() == "{}"


def test_the_prefix_is_configurable(tmp_path, monkeypatch, s3):
    monkeypatch.setenv(BUCKET_ENV, "bucket")
    monkeypatch.setenv(PREFIX_ENV, "custom/place")
    s3.objects["custom/place/repo-abc/latest.tar"] = build_tarball({"a": "b"})

    assert seed_cache_from_object_storage(tmp_path / "repo-abc") is True


def test_an_existing_cache_is_left_alone(tmp_path, monkeypatch, s3):
    """It is already a working clone, and a reader may be inside it."""
    monkeypatch.setenv(BUCKET_ENV, "bucket")
    destination = tmp_path / "repo-abc"
    destination.mkdir()
    (destination / "marker").write_text("mine")

    assert seed_cache_from_object_storage(destination) is False
    assert (destination / "marker").read_text() == "mine"
    assert s3.requested == []


def test_nothing_published_is_not_an_error(tmp_path, monkeypatch, s3):
    monkeypatch.setenv(BUCKET_ENV, "bucket")
    assert seed_cache_from_object_storage(tmp_path / "repo-abc") is False
    assert not (tmp_path / "repo-abc").exists()


def test_a_tarball_escaping_the_cache_is_refused(tmp_path, monkeypatch, s3):
    """Object storage is shared, so a tarball is untrusted input."""
    monkeypatch.setenv(BUCKET_ENV, "bucket")
    outside = tmp_path / "outside.txt"
    outside.write_text("original")
    s3.objects["artifacts/plugins/repo-abc/latest.tar"] = build_tarball(
        {"../../outside.txt": "overwritten"}
    )

    destination = tmp_path / "nested" / "repo-abc"
    assert seed_cache_from_object_storage(destination) is False
    assert outside.read_text() == "original"
    assert not destination.exists()


def test_a_broken_tarball_leaves_no_partial_cache(tmp_path, monkeypatch, s3):
    """A half-written tree would be mistaken for a complete clone."""
    monkeypatch.setenv(BUCKET_ENV, "bucket")
    s3.objects["artifacts/plugins/repo-abc/latest.tar"] = b"not a tar at all"

    destination = tmp_path / "repo-abc"
    assert seed_cache_from_object_storage(destination) is False
    assert not destination.exists()


def test_the_directory_name_comes_from_the_cache_path(tmp_path, monkeypatch, s3):
    """The key is derived from the same name get_cache_path produces."""
    monkeypatch.setenv(BUCKET_ENV, "bucket")
    seed_cache_from_object_storage(Path(tmp_path) / "claude-plugins-d078a207")

    assert s3.requested == ["artifacts/plugins/claude-plugins-d078a207/latest.tar"]
