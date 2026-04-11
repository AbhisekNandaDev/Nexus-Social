"""
Analyse Locust CSV output and print a formatted SLA report.

Usage:
    python tests/load/analyze_results.py tests/load/results/run_stats.csv

The script reads the _stats.csv file produced by:
    locust ... --csv tests/load/results/run

Columns used from Locust stats CSV:
    Name, Request Count, Failure Count, 50%, 95%, 99%, Average Response Time,
    Max Response Time, Requests/s, Failures/s
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

# ── SLA thresholds (milliseconds at p95) ──────────────────────────────────────
# Adjust these as the API matures and you have baseline measurements.
SLA_P95_MS: dict[str, int] = {
    "POST /auth/register":         800,   # bcrypt is expensive — generous threshold
    "POST /auth/login":            300,   # bcrypt + DB + Redis write
    "GET /auth/me":                 80,   # Redis blacklist check + DB point-read
    "POST /auth/refresh":          150,   # Redis del + DB revoke + DB insert + Redis set
    "POST /auth/logout":           150,   # Redis set (blacklist) + DB update + Redis del
    "POST /auth/logout [teardown]": 150,
    "POST /auth/register [fallback]": 800,
    "POST /auth/refresh [retry]":  150,
}
DEFAULT_SLA_P95_MS = 500  # fallback for endpoints not in the map above

# ANSI colours
GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW = "\033[93m"
BOLD  = "\033[1m"
RESET = "\033[0m"


@dataclass
class Row:
    name: str
    method: str
    requests: int
    failures: int
    p50: float
    p95: float
    p99: float
    avg: float
    max_: float
    rps: float
    fail_rate: float  # percentage

    @property
    def sla_threshold(self) -> int:
        key = f"{self.method} {self.name}"
        return SLA_P95_MS.get(key, DEFAULT_SLA_P95_MS)

    @property
    def sla_ok(self) -> bool:
        return self.p95 <= self.sla_threshold

    @property
    def error_rate_ok(self) -> bool:
        return self.fail_rate < 1.0  # < 1% failures


def _ms(val: str) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _pct(failures: int, total: int) -> float:
    return (failures / total * 100) if total > 0 else 0.0


def parse_csv(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("Name") in ("Aggregated", ""):
                continue
            total = int(r.get("Request Count", 0) or 0)
            failures = int(r.get("Failure Count", 0) or 0)
            rows.append(Row(
                name=r.get("Name", ""),
                method=r.get("Type", ""),
                requests=total,
                failures=failures,
                p50=_ms(r.get("50%")),
                p95=_ms(r.get("95%")),
                p99=_ms(r.get("99%")),
                avg=_ms(r.get("Average Response Time")),
                max_=_ms(r.get("Max Response Time")),
                rps=_ms(r.get("Requests/s")),
                fail_rate=_pct(failures, total),
            ))
    return sorted(rows, key=lambda r: r.name)


def _colour_ms(value: float, threshold: int) -> str:
    colour = GREEN if value <= threshold else RED
    return f"{colour}{value:>6.0f}ms{RESET}"


def _colour_pct(value: float) -> str:
    colour = GREEN if value < 1.0 else (YELLOW if value < 5.0 else RED)
    return f"{colour}{value:>5.1f}%{RESET}"


def _sla_badge(ok: bool) -> str:
    return f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"


def print_report(rows: list[Row], csv_path: Path) -> None:
    print(f"\n{BOLD}Load Test Results — {csv_path.stem}{RESET}")
    print(f"{'─' * 110}")

    header = (
        f"{'Endpoint':<38} {'Reqs':>6} {'Err%':>7} "
        f"{'p50':>8} {'p95':>8} {'p99':>8} {'avg':>8} {'max':>8} "
        f"{'RPS':>7}  {'SLA p95':>8}"
    )
    print(f"{BOLD}{header}{RESET}")
    print(f"{'─' * 110}")

    all_pass = True
    for row in rows:
        endpoint = f"{row.method} {row.name}"
        sla = row.sla_threshold

        p95_str = _colour_ms(row.p95, sla)
        err_str = _colour_pct(row.fail_rate)
        badge   = _sla_badge(row.sla_ok and row.error_rate_ok)
        if not (row.sla_ok and row.error_rate_ok):
            all_pass = False

        print(
            f"{endpoint:<38} {row.requests:>6,} {err_str} "
            f"{row.p50:>6.0f}ms {p95_str} {row.p99:>6.0f}ms "
            f"{row.avg:>6.0f}ms {row.max_:>6.0f}ms "
            f"{row.rps:>7.1f}  {badge} (<{sla}ms)"
        )

    print(f"{'─' * 110}")
    overall = f"{GREEN}{BOLD}ALL PASS{RESET}" if all_pass else f"{RED}{BOLD}FAILURES DETECTED{RESET}"
    print(f"Overall: {overall}\n")

    if not all_pass:
        print(f"{YELLOW}Endpoints that missed SLA:{RESET}")
        for row in rows:
            if not row.sla_ok:
                print(f"  {row.method} {row.name}: p95={row.p95:.0f}ms  threshold={row.sla_threshold}ms")
            if not row.error_rate_ok:
                print(f"  {row.method} {row.name}: error_rate={row.fail_rate:.1f}%  threshold=1%")
        print()


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path/to/run_stats.csv>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)

    rows = parse_csv(path)
    if not rows:
        print("No data rows found in CSV (only Aggregated row?). Did the test run long enough?")
        sys.exit(1)

    print_report(rows, path)


if __name__ == "__main__":
    main()
