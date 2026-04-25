"""TOTP 2FA gate for the dashboard, plus API keys for machine access.

Every request outside the explicit allow-list must authenticate via one of:

  * Session cookie — set by /auth/login after a valid TOTP code. Browser flow.
  * API key — `Authorization: Bearer <key>`. Intended for scripts, curl, CI,
    or the on-box agent. Keys are created from the /auth/api-keys page
    (itself gated by the session cookie, so a compromised key can't mint
    more keys).

HTML clients that fail auth are redirected to /auth/login; API/SSE clients
get a 401 JSON body.

State lives at $DASHBOARD_AUTH_FILE (default /workspace/dashboard-auth.json,
0600). To reset after losing an authenticator device: delete that file and
reload the dashboard — the setup flow starts over.

Env overrides:
    DASHBOARD_AUTH_FILE            path to the state JSON (default /workspace/dashboard-auth.json)
    DASHBOARD_SESSION_TTL_SECONDS  session cookie lifetime (default 86400 = 24h)
    DASHBOARD_AUTH_ISSUER          issuer shown in authenticator app (default "ML Stack Dashboard")
    DASHBOARD_AUTH_ACCOUNT         account label in the authenticator (default "admin")
    DASHBOARD_SESSION_SECURE       "1" → set Secure flag on session cookie
                                   (required when served over https only)
"""
from __future__ import annotations

import base64
import hmac
import hashlib
import io
import json
import os
import secrets
import time
from pathlib import Path
from typing import Optional

import pyotp
import qrcode
from qrcode.image.svg import SvgPathImage

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware


AUTH_FILE = Path(os.environ.get("DASHBOARD_AUTH_FILE", "/workspace/dashboard-auth.json"))
SESSION_TTL = int(os.environ.get("DASHBOARD_SESSION_TTL_SECONDS", str(24 * 3600)))
SESSION_SECURE = os.environ.get("DASHBOARD_SESSION_SECURE", "0") == "1"
COOKIE_NAME = "ml_dash_session"
ISSUER = os.environ.get("DASHBOARD_AUTH_ISSUER", "ML Stack Dashboard")
ACCOUNT = os.environ.get("DASHBOARD_AUTH_ACCOUNT", "admin")

# Shipped raw key looks like `mlsk_<43 url-safe chars>`. The prefix gives a
# cheap grep signal if one ever shows up in logs / git / bug reports.
API_KEY_PREFIX = "mlsk_"

# Paths that bypass the gate: the auth pages themselves, plus static assets
# referenced from the login HTML.
_PUBLIC_PREFIXES = ("/static/",)
_PUBLIC_EXACT = {
    "/auth/login",
    "/auth/setup",
    "/auth/logout",
    "/favicon.ico",
}

# Paths that must only be reachable via the session cookie — API keys are
# explicitly forbidden here, so a leaked key can't create more keys or
# enumerate the existing ones.
_SESSION_ONLY_PREFIXES = ("/auth/api-keys",)


# ── state file ────────────────────────────────────────────────────────────

def _load_state() -> dict:
    if AUTH_FILE.exists():
        try:
            return json.loads(AUTH_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_state(state: dict) -> None:
    AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = AUTH_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    tmp.replace(AUTH_FILE)


def _ensure_hmac_key(state: dict) -> bytes:
    if "hmac_key" not in state:
        state["hmac_key"] = secrets.token_urlsafe(32)
        _save_state(state)
    return state["hmac_key"].encode()


# ── API keys ──────────────────────────────────────────────────────────────

def _hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def _new_api_key() -> str:
    # 32 bytes → 43 url-safe chars (no padding). Combined with the prefix
    # this gives >256 bits of entropy, well beyond any useful brute force.
    return f"{API_KEY_PREFIX}{secrets.token_urlsafe(32)}"


def _match_api_key(state: dict, raw: str) -> Optional[dict]:
    if not raw or not raw.startswith(API_KEY_PREFIX):
        return None
    target = _hash_api_key(raw)
    for entry in state.get("api_keys", []):
        # compare_digest guards against length-leaking side channels; not
        # strictly required on sha256 hashes but cheap and correct.
        if hmac.compare_digest(entry.get("hash", ""), target):
            return entry
    return None


def _touch_api_key(state: dict, key_id: str) -> None:
    for entry in state.get("api_keys", []):
        if entry.get("id") == key_id:
            entry["last_used_at"] = int(time.time())
            _save_state(state)
            return


def _add_api_key(state: dict, label: str) -> tuple[str, dict]:
    raw = _new_api_key()
    entry = {
        "id": secrets.token_hex(4),           # short handle for the UI
        "label": label[:64] or "unnamed",
        "hash": _hash_api_key(raw),
        "prefix": raw[: len(API_KEY_PREFIX) + 4],  # e.g. "mlsk_aB3x" for display
        "created_at": int(time.time()),
        "last_used_at": None,
    }
    state.setdefault("api_keys", []).append(entry)
    _save_state(state)
    return raw, entry


def _remove_api_key(state: dict, key_id: str) -> bool:
    keys = state.get("api_keys", [])
    new_keys = [k for k in keys if k.get("id") != key_id]
    if len(new_keys) == len(keys):
        return False
    state["api_keys"] = new_keys
    _save_state(state)
    return True


def _extract_bearer(req: Request) -> Optional[str]:
    auth = req.headers.get("authorization", "")
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip() or None
    return None


def _ensure_totp_secret(state: dict) -> str:
    # First visit to /auth/setup: generate a secret and persist it so
    # reloading the page during setup keeps the same secret (otherwise the
    # authenticator app would show a stale code and confuse everyone).
    if "totp_secret" not in state:
        state["totp_secret"] = pyotp.random_base32()
        state["totp_confirmed"] = False
        _save_state(state)
    return state["totp_secret"]


# ── session cookie ────────────────────────────────────────────────────────

def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64url_dec(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sign_session(key: bytes, exp: int) -> str:
    body = _b64url(json.dumps({"exp": exp}, separators=(",", ":")).encode())
    sig = _b64url(hmac.new(key, body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def _verify_session(key: bytes, token: str) -> bool:
    try:
        body, sig = token.split(".", 1)
        expected = _b64url(hmac.new(key, body.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(expected, sig):
            return False
        payload = json.loads(_b64url_dec(body))
        return int(payload.get("exp", 0)) > int(time.time())
    except Exception:
        return False


def _issue_cookie(response: Response, key: bytes) -> None:
    exp = int(time.time()) + SESSION_TTL
    response.set_cookie(
        COOKIE_NAME,
        _sign_session(key, exp),
        max_age=SESSION_TTL,
        httponly=True,
        samesite="lax",
        secure=SESSION_SECURE,
        path="/",
    )


# ── QR rendering (SVG, no Pillow needed) ──────────────────────────────────

def _qr_svg(uri: str) -> str:
    img = qrcode.make(uri, image_factory=SvgPathImage)
    buf = io.BytesIO()
    img.save(buf)
    return buf.getvalue().decode()


# ── request classification ────────────────────────────────────────────────

def _is_public(path: str) -> bool:
    if path in _PUBLIC_EXACT:
        return True
    return any(path.startswith(p) for p in _PUBLIC_PREFIXES)


def _is_session_only(path: str) -> bool:
    return any(path.startswith(p) for p in _SESSION_ONLY_PREFIXES)


def _wants_html(req: Request) -> bool:
    return "text/html" in req.headers.get("accept", "")


class _AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if _is_public(path):
            return await call_next(request)

        state = _load_state()
        hmac_key = _ensure_hmac_key(state)

        # Session cookie wins: anything reachable via cookie is fair game,
        # including the API-key management page.
        token = request.cookies.get(COOKIE_NAME, "")
        if token and _verify_session(hmac_key, token):
            return await call_next(request)

        # API keys work for /api/* and other non-management paths but are
        # deliberately rejected at /auth/api-keys so a leaked key cannot
        # escalate into minting more keys.
        bearer = _extract_bearer(request)
        if bearer and not _is_session_only(path):
            entry = _match_api_key(state, bearer)
            if entry:
                _touch_api_key(state, entry["id"])
                return await call_next(request)

        target = "/auth/login" if state.get("totp_confirmed") else "/auth/setup"
        if _wants_html(request):
            return RedirectResponse(target, status_code=303)
        return JSONResponse(
            {"error": "unauthenticated", "login_url": target},
            status_code=401,
        )


# ── HTML templates (inline — shipping another static file for two pages
# isn't worth it) ─────────────────────────────────────────────────────────

_PAGE_CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:#0f1419;color:#e6e1e5;
     display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0}
.card{background:#1d1b20;padding:32px;border-radius:12px;box-shadow:0 4px 24px rgba(0,0,0,.4);
     max-width:640px;width:90%}
h1{margin:0 0 12px;font-size:22px}
h2{margin:24px 0 8px;font-size:16px;color:#cac4d0}
p{color:#cac4d0;line-height:1.5;font-size:14px}
input[type=text]{width:100%;box-sizing:border-box;padding:12px;font-size:16px;
     background:#2b2930;color:#e6e1e5;border:1px solid #49454f;border-radius:6px}
input[name=code]{font-size:20px;letter-spacing:4px;text-align:center}
input:focus{outline:2px solid #7c4dff;border-color:transparent}
button{margin-top:16px;width:100%;padding:12px;font-size:15px;border:0;border-radius:6px;
     background:#7c4dff;color:#fff;cursor:pointer}
button:hover{background:#9574ff}
button.danger{background:#52333a;color:#f48fb1;margin-top:0;width:auto;padding:6px 12px;font-size:13px}
button.danger:hover{background:#6b3a44}
.code{font-family:ui-monospace,Consolas,monospace;background:#2b2930;padding:10px;
     border-radius:4px;word-break:break-all;font-size:13px;color:#ffd8e4}
.err{color:#f48fb1;margin-top:8px;font-size:13px}
.ok{color:#a5d6a7;margin-top:8px;font-size:13px}
.qr{display:flex;justify-content:center;margin:16px 0}
.qr svg{width:220px;height:220px;background:#fff;padding:12px;border-radius:6px}
.hint{font-size:12px;margin-top:20px;color:#938f99}
.hint code{background:#2b2930;padding:2px 6px;border-radius:3px}
.keys{list-style:none;padding:0;margin:8px 0}
.keys li{display:flex;align-items:center;justify-content:space-between;padding:10px;
     background:#2b2930;border-radius:6px;margin-bottom:6px;gap:12px}
.keys .meta{font-size:12px;color:#938f99;font-family:ui-monospace,Consolas,monospace}
.keys .label{font-weight:500}
.new-key{background:#2b2930;border:1px solid #7c4dff;padding:12px;border-radius:6px;margin-top:8px}
.new-key strong{color:#ffd8e4}
.row{display:flex;gap:8px}
.row input{flex:1}
.row button{margin-top:0;width:auto;padding:0 16px}
a{color:#d0bcff}
"""


def _render(body: str, error: str = "") -> str:
    err = f'<div class="err">{error}</div>' if error else ""
    return (
        "<!doctype html><html><head><meta charset=utf-8>"
        "<title>ML Stack — Sign in</title>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<style>{_PAGE_CSS}</style></head><body>"
        f'<div class=card>{body}{err}</div></body></html>'
    )


def _login_body() -> str:
    return (
        "<h1>Sign in</h1>"
        "<p>Enter the 6-digit code from your authenticator app.</p>"
        '<form method=post action=/auth/login>'
        '<input type=text name=code placeholder="000000" inputmode=numeric '
        'pattern="[0-9]{6}" required autofocus autocomplete=one-time-code>'
        '<button type=submit>Sign in</button></form>'
    )


def _fmt_ts(ts: Optional[int]) -> str:
    if not ts:
        return "never"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _api_keys_body(keys: list[dict], new_key_raw: Optional[str] = None,
                   message: str = "") -> str:
    rows = []
    for k in keys:
        rows.append(
            '<li><div>'
            f'<div class=label>{k.get("label","")}</div>'
            f'<div class=meta>{k.get("prefix","")}… · id {k.get("id","")} · '
            f'created {_fmt_ts(k.get("created_at"))} · '
            f'last used {_fmt_ts(k.get("last_used_at"))}</div>'
            '</div>'
            '<form method=post action=/auth/api-keys/delete style="margin:0">'
            f'<input type=hidden name=id value="{k.get("id","")}">'
            '<button class=danger type=submit>Revoke</button></form></li>'
        )
    keys_html = (
        f'<ul class=keys>{"".join(rows)}</ul>' if rows
        else '<p class=hint>No API keys yet. Create one below.</p>'
    )

    reveal = ""
    if new_key_raw:
        reveal = (
            '<div class=new-key>'
            '<p><strong>Copy this key now — it will not be shown again.</strong></p>'
            f'<div class=code>{new_key_raw}</div>'
            "<p class=hint>Use it as <code>Authorization: Bearer &lt;key&gt;</code>.</p>"
            '</div>'
        )

    msg = f'<div class=ok>{message}</div>' if message else ''

    return (
        "<h1>API keys</h1>"
        "<p>Use an API key to call <code>/api/*</code> from scripts, curl, "
        "or CI without the TOTP login flow. "
        "Send it as <code>Authorization: Bearer &lt;key&gt;</code>.</p>"
        f'{msg}{reveal}'
        "<h2>Existing keys</h2>"
        f"{keys_html}"
        "<h2>Create a new key</h2>"
        '<form method=post action=/auth/api-keys class=row>'
        '<input type=text name=label placeholder="label (e.g. agent, ci)" '
        'maxlength=64 required>'
        '<button type=submit>Create</button></form>'
        '<p class=hint><a href="/">← back to dashboard</a> · '
        '<a href="/auth/logout">sign out</a></p>'
    )


def _setup_body(secret: str) -> str:
    uri = pyotp.TOTP(secret).provisioning_uri(name=ACCOUNT, issuer_name=ISSUER)
    qr = _qr_svg(uri)
    return (
        "<h1>Set up 2FA</h1>"
        "<p>Scan this QR code with an authenticator app (Google Authenticator, "
        "Authy, 1Password), then enter the current 6-digit code to confirm.</p>"
        f'<div class=qr>{qr}</div>'
        "<p>Or enter this secret manually:</p>"
        f'<div class=code>{secret}</div>'
        '<form method=post action=/auth/setup>'
        '<input type=text name=code placeholder="000000" inputmode=numeric '
        'pattern="[0-9]{6}" required autofocus autocomplete=one-time-code>'
        '<button type=submit>Confirm</button></form>'
        f'<p class=hint>Lost your authenticator? Run <code>rm {AUTH_FILE}</code> '
        "on the host and reload this page to start over.</p>"
    )


# ── wiring ────────────────────────────────────────────────────────────────

def install(app: FastAPI) -> None:
    """Attach the 2FA middleware and auth routes to *app*.

    Call once, immediately after ``app = FastAPI(...)`` so every other route
    is registered behind the gate.
    """
    app.add_middleware(_AuthMiddleware)

    @app.get("/auth/setup", response_class=HTMLResponse)
    def _setup_page() -> HTMLResponse:
        state = _load_state()
        if state.get("totp_confirmed"):
            return RedirectResponse("/auth/login", status_code=303)
        _ensure_hmac_key(state)
        secret = _ensure_totp_secret(state)
        return HTMLResponse(_render(_setup_body(secret)))

    @app.post("/auth/setup", response_class=HTMLResponse)
    def _setup_submit(code: str = Form(...)):
        state = _load_state()
        secret = state.get("totp_secret")
        if not secret:
            return RedirectResponse("/auth/setup", status_code=303)
        if not pyotp.TOTP(secret).verify(code.strip(), valid_window=1):
            return HTMLResponse(_render(_setup_body(secret), "Wrong code, try again."))
        state["totp_confirmed"] = True
        _save_state(state)
        key = _ensure_hmac_key(state)
        resp = RedirectResponse("/", status_code=303)
        _issue_cookie(resp, key)
        return resp

    @app.get("/auth/login", response_class=HTMLResponse)
    def _login_page() -> HTMLResponse:
        state = _load_state()
        if not state.get("totp_confirmed"):
            return RedirectResponse("/auth/setup", status_code=303)
        return HTMLResponse(_render(_login_body()))

    @app.post("/auth/login", response_class=HTMLResponse)
    def _login_submit(code: str = Form(...)):
        state = _load_state()
        secret = state.get("totp_secret")
        if not secret or not state.get("totp_confirmed"):
            return RedirectResponse("/auth/setup", status_code=303)
        if not pyotp.TOTP(secret).verify(code.strip(), valid_window=1):
            return HTMLResponse(_render(_login_body(), "Wrong code, try again."))
        key = _ensure_hmac_key(state)
        resp = RedirectResponse("/", status_code=303)
        _issue_cookie(resp, key)
        return resp

    @app.post("/auth/logout")
    def _logout():
        resp = RedirectResponse("/auth/login", status_code=303)
        resp.delete_cookie(COOKIE_NAME, path="/")
        return resp

    @app.get("/auth/logout")
    def _logout_get():
        resp = RedirectResponse("/auth/login", status_code=303)
        resp.delete_cookie(COOKIE_NAME, path="/")
        return resp

    # ── API keys ──────────────────────────────────────────────────────────
    # These routes sit behind the session-cookie gate (see _SESSION_ONLY_PREFIXES)
    # so an API key can never be used to create or enumerate other keys.

    @app.get("/auth/api-keys", response_class=HTMLResponse)
    def _api_keys_page() -> HTMLResponse:
        state = _load_state()
        return HTMLResponse(_render(_api_keys_body(_list_api_keys(state))))

    @app.post("/auth/api-keys", response_class=HTMLResponse)
    def _api_keys_create(label: str = Form(...)):
        state = _load_state()
        raw, entry = _add_api_key(state, label.strip())
        return HTMLResponse(_render(
            _api_keys_body(_list_api_keys(state), new_key_raw=raw,
                           message=f'Created "{entry["label"]}".')
        ))

    @app.post("/auth/api-keys/delete", response_class=HTMLResponse)
    def _api_keys_delete(id: str = Form(...)):
        state = _load_state()
        removed = _remove_api_key(state, id.strip())
        msg = "Key revoked." if removed else "No such key."
        return HTMLResponse(_render(
            _api_keys_body(_list_api_keys(state), message=msg)
        ))


def _list_api_keys(state: dict) -> list[dict]:
    """Public listing (never exposes the hash)."""
    out = []
    for k in state.get("api_keys", []):
        out.append({
            "id": k.get("id", ""),
            "label": k.get("label", ""),
            "prefix": k.get("prefix", ""),
            "created_at": k.get("created_at"),
            "last_used_at": k.get("last_used_at"),
        })
    return out
