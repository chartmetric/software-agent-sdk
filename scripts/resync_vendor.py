#!/usr/bin/env python
"""Follow the upstreams we vendor a file from, instead of drifting from them.

A vendored file is a copy of somebody else's code taken at a commit. Nothing
reminds you when they fix it, so this does: `--check` fetches the pinned file
and reports whether our copy still matches, and whether upstream's default
branch has moved past the pin. `--update` takes their current file and rewrites
the pin.

Each vendored directory carries a `VENDOR.json` naming the repository, the
commit, the license and, per file, its upstream path and sha256.

    python scripts/resync_vendor.py --check
    python scripts/resync_vendor.py --update --only jev-ultrafast
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RAW = "https://raw.githubusercontent.com/{owner_repo}/{ref}/{path}"
API_BRANCH = "https://api.github.com/repos/{owner_repo}/commits/HEAD"
TIMEOUT_SECONDS = 30


def owner_repo(repository: str) -> str:
    return repository.rstrip("/").removeprefix("https://github.com/")


def fetch(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            return response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"  ! could not fetch {url}: {exc}")
        return None


def head_commit(repository: str) -> str | None:
    body = fetch(API_BRANCH.format(owner_repo=owner_repo(repository)))
    if body is None:
        return None
    try:
        return json.loads(body)["sha"]
    except (ValueError, KeyError):
        return None


def check(manifest_path: Path, update: bool) -> bool:
    manifest = json.loads(manifest_path.read_text())
    repository, commit = manifest["repository"], manifest["commit"]
    print(f"{manifest_path.parent.name}: {repository} @ {commit[:9]}")
    settled = True
    for name, entry in manifest["files"].items():
        local = manifest_path.parent / name
        ours = hashlib.sha256(local.read_bytes()).hexdigest()
        if ours != entry["sha256"]:
            print(
                f"  ! {name} has been edited in place (sha256 {ours[:12]}…); "
                "a vendored file is a copy, so change ours and re-pin instead"
            )
            settled = False
        pinned = fetch(
            RAW.format(
                owner_repo=owner_repo(repository),
                ref=commit,
                path=entry["upstream_path"],
            )
        )
        if pinned is None:
            settled = False
            continue
        if hashlib.sha256(pinned).hexdigest() != entry["sha256"]:
            print(f"  ! {name} does not match upstream at the pinned commit")
            settled = False
    head = head_commit(repository)
    if head and head != commit:
        print(f"  upstream has moved to {head[:9]}; read their diff before ours")
        if update:
            for name, entry in manifest["files"].items():
                latest = fetch(
                    RAW.format(
                        owner_repo=owner_repo(repository),
                        ref=head,
                        path=entry["upstream_path"],
                    )
                )
                if latest is None:
                    return False
                (manifest_path.parent / name).write_bytes(latest)
                entry["sha256"] = hashlib.sha256(latest).hexdigest()
            manifest["commit"] = head
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(
                "  updated the copy and the pin; review the diff and our own "
                "code against it"
            )
            return True
    elif head:
        print("  in step with upstream")
    return settled


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--only", default=None)
    arguments = parser.parse_args()
    if not (arguments.check or arguments.update):
        parser.error("pass --check or --update")
    manifests = sorted(ROOT.glob("*/openhands/tools/*/vendor/*/VENDOR.json"))
    if arguments.only:
        manifests = [m for m in manifests if m.parent.name == arguments.only]
    if not manifests:
        print("no vendored directories found")
        return 1
    return 0 if all(check(m, arguments.update) for m in manifests) else 1


if __name__ == "__main__":
    sys.exit(main())
