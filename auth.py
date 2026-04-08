# ─── auth.py ──────────────────────────────────────────────────────────────────
"""Simple JSON-file based user authentication."""

import json, os, hashlib
from datetime import datetime

USERS_FILE = "data/users.json"


def _load() -> list[dict]:
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return []


def _save(users: list[dict]):
    os.makedirs("data", exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def register(username: str, email: str, password: str) -> dict:
    users = _load()
    if any(u["username"].lower() == username.lower() for u in users):
        return {"ok": False, "msg": "Username already exists."}
    if any(u["email"].lower() == email.lower() for u in users):
        return {"ok": False, "msg": "Email already registered."}
    if len(password) < 6:
        return {"ok": False, "msg": "Password must be at least 6 characters."}
    users.append({
        "username": username,
        "email": email,
        "password": _hash(password),
        "joined": datetime.now().strftime("%Y-%m-%d"),
    })
    _save(users)
    return {"ok": True, "msg": "Account created! Please log in."}


def login(username: str, password: str) -> dict:
    users = _load()
    for u in users:
        if u["username"].lower() == username.lower() and u["password"] == _hash(password):
            return {"ok": True, "user": {"username": u["username"], "email": u["email"]}}
    return {"ok": False, "msg": "Invalid username or password."}
