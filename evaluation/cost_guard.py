"""Persistent conservative experiment budget, not an AWS account billing cap."""

import json
import math
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from ml.tune_models import atomic_json

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None
    import msvcrt

PRICING_SOURCE = (
    "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-terra.html"
)
# GovCloud short-context standard rates, checked 2026-09-08. Reserve at the
# higher cache-write input rate, ignoring cache savings entirely.
INPUT_PER_MILLION = 3.30
OUTPUT_PER_MILLION = 15.84
DEFAULT_LIMIT_USD = 10.0
HARD_MAX_LIMIT_USD = 10.0
DEFAULT_MAX_REQUESTS = 96
HARD_MAX_REQUESTS = 512
MAX_INPUT_BYTES = 200_000
MAX_OUTPUT_TOKENS = 6_000


class BudgetExceeded(RuntimeError):
    pass


@contextmanager
def exclusive_lock(path: Path):
    with path.open("a+b") as lock:
        lock.seek(0, 2)
        if lock.tell() == 0:
            lock.write(b"\0")
            lock.flush()
        lock.seek(0)
        if fcntl is not None:
            fcntl.flock(lock, fcntl.LOCK_EX)
        else:
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            lock.seek(0)
            if fcntl is not None:
                fcntl.flock(lock, fcntl.LOCK_UN)
            else:
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def cost(input_tokens, output_tokens):
    for name, value in (("input_tokens", input_tokens), ("output_tokens", output_tokens)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    return (
        math.ceil(input_tokens * INPUT_PER_MILLION + output_tokens * OUTPUT_PER_MILLION) / 1_000_000
    )


class CostGuard:
    def __init__(
        self,
        path: Path,
        limit_usd=DEFAULT_LIMIT_USD,
        max_requests=DEFAULT_MAX_REQUESTS,
    ):
        if (
            not isinstance(limit_usd, (int, float))
            or isinstance(limit_usd, bool)
            or not math.isfinite(limit_usd)
            or not 0 < limit_usd <= HARD_MAX_LIMIT_USD
        ):
            raise ValueError(f"Budget must be above $0 and at most ${HARD_MAX_LIMIT_USD:g}")
        if (
            not isinstance(max_requests, int)
            or isinstance(max_requests, bool)
            or not 1 <= max_requests <= HARD_MAX_REQUESTS
        ):
            raise ValueError(
                f"Request limit must be an integer from 1 through {HARD_MAX_REQUESTS}"
            )
        self.path = Path(path)
        self.limit = float(limit_usd)
        self.max_requests = max_requests
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self):
        if self.path.exists():
            ledger = json.loads(self.path.read_text())
            if not isinstance(ledger, dict) or not isinstance(ledger.get("requests"), list):
                raise ValueError("Malformed cost ledger")
            # Ledgers created before request limits were configurable used 96.
            stored_max_requests = ledger.get("max_requests", DEFAULT_MAX_REQUESTS)
            stored_limit = ledger.get("limit_usd")
            if (
                not isinstance(stored_limit, (int, float))
                or isinstance(stored_limit, bool)
                or not math.isfinite(stored_limit)
                or not isinstance(stored_max_requests, int)
                or isinstance(stored_max_requests, bool)
                or not 1 <= stored_max_requests <= HARD_MAX_REQUESTS
            ):
                raise ValueError("Malformed cost ledger limits")
            if (
                stored_limit != self.limit
                or stored_max_requests != self.max_requests
            ):
                raise ValueError("Cannot change the budget or request limit of an existing run")
            for record in ledger["requests"]:
                accounted = record.get("accounted_usd")
                if (
                    not isinstance(accounted, (int, float))
                    or isinstance(accounted, bool)
                    or not math.isfinite(accounted)
                    or accounted < 0
                ):
                    raise ValueError("Malformed cost ledger request")
            accounted = ledger.get("accounted_usd")
            request_total = sum(record["accounted_usd"] for record in ledger["requests"])
            if (
                not isinstance(accounted, (int, float))
                or isinstance(accounted, bool)
                or not math.isfinite(accounted)
                or not math.isclose(accounted, request_total, rel_tol=0, abs_tol=1e-12)
            ):
                raise ValueError("Cost ledger total does not match its requests")
            ledger.setdefault("max_requests", stored_max_requests)
            return ledger
        return {
            "limit_usd": self.limit,
            "max_requests": self.max_requests,
            "hard_max_limit_usd": HARD_MAX_LIMIT_USD,
            "hard_max_requests": HARD_MAX_REQUESTS,
            "pricing_source": PRICING_SOURCE,
            "pricing_checked": "2026-09-08",
            "input_rate": INPUT_PER_MILLION,
            "output_rate": OUTPUT_PER_MILLION,
            "requests": [],
            "accounted_usd": 0.0,
            "scope": "This evaluation only; excludes taxes and other AWS resources",
        }

    def request(self, delegate, prompt, max_output_tokens):
        if (
            not isinstance(max_output_tokens, int)
            or isinstance(max_output_tokens, bool)
            or not 1 <= max_output_tokens <= MAX_OUTPUT_TOKENS
        ):
            raise ValueError("Output exceeds approved per-request token bound")
        # Byte count deliberately overestimates text BPE tokens; headroom covers
        # the Responses envelope. Large contexts are rejected, not repriced.
        input_bound = len(json.dumps(prompt, separators=(",", ":")).encode()) + 1024
        if input_bound > MAX_INPUT_BYTES:
            raise ValueError("Request exceeds the short-context budget guard")
        reserve = cost(input_bound, max_output_tokens)
        with exclusive_lock(self.path.with_suffix(".lock")):
            ledger = self.load()
            spent = sum(r["accounted_usd"] for r in ledger["requests"])
            if spent + reserve > self.limit or len(ledger["requests"]) >= self.max_requests:
                raise BudgetExceeded(
                    "Next request could exceed the configured "
                    f"${self.limit:g} or {self.max_requests}-attempt limit"
                )
            record = {
                "started_at": datetime.now(UTC).isoformat(),
                "reserved_usd": reserve,
                "accounted_usd": reserve,
                "input_bound": input_bound,
                "max_output_tokens": max_output_tokens,
                "state": "reserved",
            }
            ledger["requests"].append(record)
            ledger["accounted_usd"] = spent + reserve
            atomic_json(self.path, ledger)
            delegate.last_usage = {}
            try:
                result = delegate.request_json(prompt, max_output_tokens=max_output_tokens)
            except Exception:
                record["state"] = "failed_or_unknown_usage_reservation_retained"
                atomic_json(self.path, ledger)
                raise
            usage = getattr(delegate, "last_usage", {})
            counts = [usage.get("input_tokens"), usage.get("output_tokens")]
            if all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in counts
            ):
                actual = cost(*counts)
                record.update(
                    {
                        "usage": usage,
                        "accounted_usd": actual,
                        "state": "accounted_at_conservative_rates",
                    }
                )
                ledger["accounted_usd"] = spent + actual
                atomic_json(self.path, ledger)
                if counts[0] > input_bound or counts[1] > max_output_tokens or actual > reserve:
                    raise BudgetExceeded(
                        "Usage exceeded the reserved bound; stop and inspect billing"
                    )
            else:
                record["state"] = "unknown_usage_reservation_retained"
                atomic_json(self.path, ledger)
            return result


class BudgetedAdapter:
    def __init__(self, delegate, guard):
        settings = delegate.settings
        if settings.bedrock_model_id != "openai.gpt-5.6-terra" or settings.aws_region not in {
            "us-gov-west-1",
            "us-gov-east-1",
        }:
            raise ValueError("Budget rates only verified for GPT-5.6 Terra in GovCloud")
        self.delegate = delegate
        self.guard = guard

    @property
    def last_usage(self):
        return getattr(self.delegate, "last_usage", {})

    def request_json(self, prompt, max_output_tokens=3000):
        return self.guard.request(self.delegate, prompt, max_output_tokens)
