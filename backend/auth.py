from __future__ import annotations
import base64, hashlib, hmac, json, time, uuid
from fastapi import Depends, Header, HTTPException
from .config import settings

def create_token(role: str) -> str:
    payload = {"role": role, "sid": uuid.uuid4().hex, "exp": int(time.time()) + 60 * 60 * 8}
    raw = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")
    sig = hmac.new(settings.session_secret.encode(), raw.encode(), hashlib.sha256).hexdigest()
    return raw + "." + sig

def current_session(authorization: str | None = Header(default=None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "A valid local session is required.")
    try:
        raw, sig = authorization[7:].split(".", 1)
        expected = hmac.new(settings.session_secret.encode(), raw.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected): raise ValueError("signature")
        payload = json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))
        if payload["exp"] < time.time() or payload["role"] not in {"student", "teacher"}: raise ValueError("expired")
        return payload
    except Exception:
        raise HTTPException(401, "Invalid or expired local session.")

def require_teacher(session: dict = Depends(current_session)) -> dict:
    if session["role"] != "teacher": raise HTTPException(403, "Teacher authorization is required.")
    return session
