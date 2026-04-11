"""
Bulk user seeding script for load testing.

Creates N users via the API concurrently and saves their credentials to a JSON
file that locustfile.py can consume so the load test starts from the login step
(not the registration step), giving a cleaner read on steady-state performance.

Usage:
    python tests/load/setup_users.py \\
        --count 1000 \\
        --base-url http://localhost:8000 \\
        --out tests/load/users.json \\
        --concurrency 50

The script is idempotent: if users.json already exists it merges new entries in.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

import httpx
from tqdm.asyncio import tqdm

BASE_PASSWORD = "LoadTest123!"
EMAIL_DOMAIN = "example.com"


async def register_user(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    base_url: str,
    uid: str,
) -> dict | None:
    email = f"load_{uid}@{EMAIL_DOMAIN}"
    payload = {
        "email": email,
        "display_name": f"PerfUser {uid[:6]}",
        "password": BASE_PASSWORD,
        "confirm_password": BASE_PASSWORD,
    }
    async with sem:
        try:
            resp = await client.post(f"{base_url}/api/v1/auth/register", json=payload, timeout=30)
            if resp.status_code == 201:
                data = resp.json()
                return {"email": email, "password": BASE_PASSWORD, "user_id": data["user_id"]}
            elif resp.status_code == 409:
                # Already exists — record credentials without a user_id (login will resolve it)
                return {"email": email, "password": BASE_PASSWORD, "user_id": None}
            else:
                return None  # unexpected failure — skip
        except (httpx.RequestError, httpx.TimeoutException):
            return None


async def main(count: int, base_url: str, out: Path, concurrency: int) -> None:
    # Load existing users so we don't blow away prior runs
    existing: list[dict] = []
    if out.exists():
        existing = json.loads(out.read_text())
        print(f"Loaded {len(existing)} existing users from {out}")

    needed = max(0, count - len(existing))
    if needed == 0:
        print(f"Already have {len(existing)} users — nothing to do.")
        return

    print(f"Registering {needed} new users (concurrency={concurrency}) …")

    sem = asyncio.Semaphore(concurrency)
    uids = [uuid.uuid4().hex[:12] for _ in range(needed)]

    async with httpx.AsyncClient() as client:
        # Connectivity check
        try:
            await client.get(f"{base_url}/health", timeout=5)
        except Exception as exc:
            print(f"ERROR: cannot reach {base_url} — {exc}", file=sys.stderr)
            sys.exit(1)

        tasks = [register_user(client, sem, base_url, uid) for uid in uids]
        results: list[dict | None] = []
        async for result in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="registering"):
            results.append(await result)

    new_users = [r for r in results if r is not None]
    failed = len(results) - len(new_users)
    if failed:
        print(f"WARNING: {failed} registrations failed (server errors / timeouts)")

    all_users = existing + new_users
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_users, indent=2))
    print(f"Saved {len(all_users)} users → {out}")
    print(f"  New: {len(new_users)}  |  Existing: {len(existing)}  |  Failed: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk-seed users for load testing")
    parser.add_argument("--count", type=int, default=500, help="Total users to ensure exist")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--out", default="tests/load/users.json", help="Output JSON file")
    parser.add_argument("--concurrency", type=int, default=50, help="Concurrent HTTP requests")
    args = parser.parse_args()

    asyncio.run(main(args.count, args.base_url, Path(args.out), args.concurrency))
