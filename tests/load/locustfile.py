"""
Load test for the Social Media Platform auth API.

Two user classes are provided — run one or both depending on what you want to measure:

  ActiveUser (default)
    Reads from tests/load/users.json (pre-seeded via setup_users.py).
    Simulates steady-state traffic: login → /me (hot path) → refresh → logout.
    Use this to measure throughput and latency under realistic load.

  RegistrationUser
    Registers a brand-new account on each spawn.
    Use this to stress the registration + bcrypt path independently.

Quick-start (steady-state, 10k concurrent users):
    locust -f tests/load/locustfile.py ActiveUser \\
        --host http://localhost:8000 \\
        --users 10000 \\
        --spawn-rate 200 \\
        --run-time 3m \\
        --headless \\
        --csv tests/load/results/run

Then analyse:
    python tests/load/analyze_results.py tests/load/results/run_stats.csv
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path

from locust import between, events, task
from locust.contrib.fasthttp import FastHttpUser

# ── User pool (pre-seeded users) ───────────────────────────────────────────────

_USERS_FILE = Path(os.environ.get("LOCUST_USERS_FILE", "tests/load/users.json"))
_user_pool: list[dict] = []
_pool_lock = threading.Lock()
_pool_index = 0


def _load_user_pool() -> None:
    global _user_pool
    if _USERS_FILE.exists():
        _user_pool = json.loads(_USERS_FILE.read_text())
        print(f"[pool] loaded {len(_user_pool)} pre-seeded users from {_USERS_FILE}")
    else:
        print(f"[pool] {_USERS_FILE} not found — ActiveUser will fall back to registration")


def _next_pool_user() -> dict | None:
    """Round-robin across the pool so each virtual user gets a unique account."""
    global _pool_index
    if not _user_pool:
        return None
    with _pool_lock:
        user = _user_pool[_pool_index % len(_user_pool)]
        _pool_index += 1
    return user


@events.init.add_listener
def on_locust_init(**_kwargs):
    _load_user_pool()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── ActiveUser ─────────────────────────────────────────────────────────────────

class ActiveUser(FastHttpUser):
    """
    Simulates a logged-in user hitting the platform.

    Task distribution (weights approximate real traffic):
      - GET /me               x10  — authenticated reads (hot path, Redis blacklist check)
      - rotate_tokens         x3   — access + refresh token rotation
      - logout + login        x1   — full session cycle

    Requires users.json. Falls back to registering a fresh account if the pool is empty.

    NOTE: instance attributes use underscore prefix (_access_token, _refresh_tok) to
    avoid shadowing task method names (which would cause 'method is not JSON serializable').
    """
    wait_time = between(0.5, 2)

    _access_token: str | None = None
    _refresh_tok: str | None = None   # NOT named refresh_token — that's a @task method
    _email: str | None = None
    _password: str | None = None

    def on_start(self) -> None:
        pool_user = _next_pool_user()
        if pool_user:
            self._email = pool_user["email"]
            self._password = pool_user["password"]
            self._do_login()
        else:
            self._register_fresh()

    # ── auth flows ──────────────────────────────────────────────────────────────

    def _do_login(self) -> None:
        with self.client.post(
            "/api/v1/auth/login",
            json={"email": self._email, "password": self._password},
            name="POST /auth/login",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                self._access_token = data["access_token"]
                self._refresh_tok = data["refresh_token"]
                resp.success()
            else:
                resp.failure(f"login failed: {resp.status_code} {(resp.text or '')[:120]}")

    def _register_fresh(self) -> None:
        uid = uuid.uuid4().hex[:10]
        self._email = f"load_{uid}@example.com"
        self._password = "LoadTest123!"
        with self.client.post(
            "/api/v1/auth/register",
            json={
                "email": self._email,
                "display_name": f"PerfUser {uid[:6]}",
                "password": self._password,
                "confirm_password": self._password,
            },
            name="POST /auth/register [fallback]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 201:
                data = resp.json()
                self._access_token = data["access_token"]
                self._refresh_tok = data["refresh_token"]
                resp.success()
            else:
                resp.failure(f"register failed: {resp.status_code} {(resp.text or '')[:120]}")

    # ── tasks ───────────────────────────────────────────────────────────────────

    @task(10)
    def get_me(self) -> None:
        """Hot path: authenticated profile read. Tests Redis blacklist check + DB user fetch."""
        if not self._access_token:
            return
        with self.client.get(
            "/api/v1/auth/me",
            headers=_auth_headers(self._access_token),
            name="GET /auth/me",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            elif resp.status_code == 401:
                resp.failure("401 on /me — token expired or revoked")
                self._try_rotate()
            else:
                resp.failure(f"unexpected {resp.status_code}")

    @task(3)
    def rotate_tokens(self) -> None:
        """Rotate access + refresh tokens. Tests Redis del + DB revoke + new DB write."""
        if not self._refresh_tok:
            return
        with self.client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": self._refresh_tok},
            name="POST /auth/refresh",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                self._access_token = data["access_token"]
                resp.success()
            elif resp.status_code == 401:
                resp.failure("refresh token invalid — re-logging in")
                self._do_login()
            else:
                resp.failure(f"unexpected {resp.status_code}")

    @task(1)
    def logout_and_login(self) -> None:
        """Full session cycle: logout (blacklists JWT + revokes refresh token) then re-login."""
        if self._access_token and self._refresh_tok:
            with self.client.post(
                "/api/v1/auth/logout",
                json={"refresh_token": self._refresh_tok},
                headers=_auth_headers(self._access_token),
                name="POST /auth/logout",
                catch_response=True,
            ) as resp:
                if resp.status_code == 200:
                    resp.success()
                else:
                    resp.failure(f"logout {resp.status_code}")
            self._access_token = None
            self._refresh_tok = None

        self._do_login()

    def _try_rotate(self) -> None:
        if not self._refresh_tok:
            return
        resp = self.client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": self._refresh_tok},
            name="POST /auth/refresh [retry]",
        )
        if resp.status_code == 200:
            self._access_token = resp.json()["access_token"]

    def on_stop(self) -> None:
        if self._access_token and self._refresh_tok:
            self.client.post(
                "/api/v1/auth/logout",
                json={"refresh_token": self._refresh_tok},
                headers=_auth_headers(self._access_token),
                name="POST /auth/logout [teardown]",
            )


# ── RegistrationUser ───────────────────────────────────────────────────────────

class RegistrationUser(FastHttpUser):
    """
    Stress-tests the registration endpoint exclusively.
    Each virtual user registers once and then hammers /me.

    Useful for isolating bcrypt cost and DB insert throughput.
    Run separately from ActiveUser to avoid mixing metrics:

        locust -f tests/load/locustfile.py RegistrationUser \\
            --host http://localhost:8000 --users 500 --spawn-rate 50
    """
    wait_time = between(1, 3)

    _access_token: str | None = None

    def on_start(self) -> None:
        uid = uuid.uuid4().hex[:10]
        with self.client.post(
            "/api/v1/auth/register",
            json={
                "email": f"reg_{uid}@example.com",
                "display_name": f"RegUser {uid[:6]}",
                "password": "LoadTest123!",
                "confirm_password": "LoadTest123!",
            },
            name="POST /auth/register",
            catch_response=True,
        ) as resp:
            if resp.status_code == 201:
                data = resp.json()
                self._access_token = data["access_token"]
                resp.success()
            else:
                resp.failure(f"{resp.status_code}: {(resp.text or '')[:120]}")

    @task
    def get_me(self) -> None:
        if not self._access_token:
            return
        self.client.get(
            "/api/v1/auth/me",
            headers=_auth_headers(self._access_token),
            name="GET /auth/me",
        )
