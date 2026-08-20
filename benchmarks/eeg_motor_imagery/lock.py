"""Hash-addressed protocol lock shared by tuning and evaluation."""

import hashlib
import json


LOCK_VERSION = 1
SEEDS = (11, 29, 47)


def canonical_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def verify_lock(lock: dict) -> None:
    claimed = lock.get("lock_sha256")
    payload = {key: value for key, value in lock.items() if key != "lock_sha256"}
    if not claimed or claimed != canonical_hash(payload):
        raise ValueError("evaluation lock hash does not match its payload")
