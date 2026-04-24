"""TOTP 2FA gate for the dashboard.

Every request outside the explicit allow-list is redirected to /auth/login
(HTML clients) or returns 401 JSON (API/SSE clients) unless it carries a
valid HMAC-signed session cookie.

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

# Paths that bypass the gate: the auth pages themselves, plus static assets
# referenced from the login HTML.
_PUBLIC_PREFIXES = ("/static/",)
_PUBLIC_EXACT = {
    "/auth/login",
    "/auth/setup",
    "/auth/logout",
    "/favicon.ico",
}


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


def _wants_html(req: Request) -> bool:
    return "text/html" in req.headers.get("accept", "")


class _AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if _is_public(request.url.path):
            return await call_next(request)

        state = _load_state()
        key = _ensure_hmac_key(state)
        token = request.cookies.get(COOKIE_NAME, "")
        if token and _verify_session(key, token):
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
     max-width:440px;width:90%}
h1{margin:0 0 12px;font-size:22px}
p{color:#cac4d0;line-height:1.5;font-size:14px}
input[type=text]{width:100%;box-sizing:border-box;padding:12px;font-size:20px;
     background:#2b2930;color:#e6e1e5;border:1px solid #49454f;border-radius:6px;
     letter-spacing:4px;text-align:center}
input:focus{outline:2px solid #7c4dff;border-color:transparent}
button{margin-top:16px;width:100%;padding:12px;font-size:15px;border:0;border-radius:6px;
     background:#7c4dff;color:#fff;cursor:pointer}
button:hover{background:#9574ff}
.code{font-family:ui-monospace,Consolas,monospace;background:#2b2930;padding:10px;
     border-radius:4px;word-break:break-all;font-size:13px;color:#ffd8e4}
.err{color:#f48fb1;margin-top:8px;font-size:13px}
.qr{display:flex;justify-content:center;margin:16px 0}
.qr svg{width:220px;height:220px;background:#fff;padding:12px;border-radius:6px}
.hint{font-size:12px;margin-top:20px;color:#938f99}
.hint code{background:#2b2930;padding:2px 6px;border-radius:3px}
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
