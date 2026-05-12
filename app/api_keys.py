import hashlib
import json
import secrets
from datetime import UTC, datetime
from pathlib import Path

from app.config import Settings


def generate_api_key(prefix: str = "ine") -> str:
    return f"{prefix}_{secrets.token_urlsafe(32)}"


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def load_api_keys(settings: Settings) -> list[dict[str, object]]:
    path = Path(settings.api_keys_file)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        return []
    return data


def save_api_keys(settings: Settings, records: list[dict[str, object]]) -> None:
    path = Path(settings.api_keys_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


def create_api_key(settings: Settings, name: str) -> tuple[str, dict[str, object]]:
    api_key = generate_api_key()
    now = datetime.now(UTC).isoformat()
    record = {
        "id": secrets.token_hex(8),
        "name": name,
        "key_hash": hash_api_key(api_key),
        "created_at": now,
        "revoked": False,
    }
    records = load_api_keys(settings)
    records.append(record)
    save_api_keys(settings, records)
    public_record = {key: value for key, value in record.items() if key != "key_hash"}
    return api_key, public_record


def list_api_keys(settings: Settings) -> list[dict[str, object]]:
    records = load_api_keys(settings)
    return [{key: value for key, value in record.items() if key != "key_hash"} for record in records]


def revoke_api_key(settings: Settings, key_id: str) -> bool:
    records = load_api_keys(settings)
    found = False
    now = datetime.now(UTC).isoformat()
    for record in records:
        if record.get("id") == key_id:
            record["revoked"] = True
            record["revoked_at"] = now
            found = True
            break
    if found:
        save_api_keys(settings, records)
    return found


def is_valid_api_key(settings: Settings, api_key: str | None) -> bool:
    if not api_key:
        return False
    if settings.api_key and secrets.compare_digest(api_key, settings.api_key):
        return True
    incoming_hash = hash_api_key(api_key)
    return any(
        not record.get("revoked") and secrets.compare_digest(str(record.get("key_hash", "")), incoming_hash)
        for record in load_api_keys(settings)
    )
