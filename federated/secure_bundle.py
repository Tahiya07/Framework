from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Dict

import torch


BUNDLE_FORMAT = "federated_lora_update_v1"


def _derive_key() -> bytes:
    material = os.environ.get("FEDERATED_UPDATE_KEY", "framework-local-fl-key-change-in-production")
    return hashlib.sha256(material.encode("utf-8")).digest()


def _xor_bytes(data: bytes, key: bytes) -> bytes:
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))


def serialize_state_dict(state: Dict[str, torch.Tensor]) -> bytes:
    buf = io.BytesIO()
    torch.save(state, buf)
    return buf.getvalue()


def deserialize_state_dict(payload: bytes) -> Dict[str, torch.Tensor]:
    try:
        return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(io.BytesIO(payload), map_location="cpu")


def pack_update(
    *,
    client_id: str,
    round_idx: int,
    role: str,
    n_samples: int,
    state: Dict[str, torch.Tensor],
    encrypt: bool = True,
) -> Dict[str, Any]:
    raw = serialize_state_dict(state)
    digest = hashlib.sha256(raw).hexdigest()
    if encrypt:
        raw = _xor_bytes(raw, _derive_key())
    payload = base64.b64encode(raw).decode("ascii")
    return {
        "format": BUNDLE_FORMAT,
        "client_id": client_id,
        "round": int(round_idx),
        "role": role,
        "n_samples": int(n_samples),
        "encrypted": bool(encrypt),
        "encryption": "xor-sha256-key",
        "sha256_plaintext": digest,
        "payload_b64": payload,
    }


def unpack_update(bundle: Dict[str, Any], *, encrypt: bool | None = None) -> Dict[str, torch.Tensor]:
    if bundle.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"unsupported bundle format: {bundle.get('format')!r}")
    use_enc = bool(bundle.get("encrypted")) if encrypt is None else encrypt
    raw = base64.b64decode(bundle["payload_b64"])
    if use_enc:
        raw = _xor_bytes(raw, _derive_key())
    expected = bundle.get("sha256_plaintext")
    if expected and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("update bundle integrity check failed (sha256 mismatch)")
    return deserialize_state_dict(raw)


def save_bundle(path: Path, bundle: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")


def load_bundle(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
