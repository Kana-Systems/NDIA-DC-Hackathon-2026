"""Persistent conservative experiment budget, not an AWS account billing cap."""

import fcntl
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from ml.tune_models import atomic_json

PRICING_SOURCE = (
    "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-terra.html"
)
# GovCloud short-context standard rates, checked 2026-09-08. Reserve at the
# higher cache-write input rate, ignoring cache savings entirely.
INPUT_PER_MILLION = 3.30
OUTPUT_PER_MILLION = 15.84


class BudgetExceeded(RuntimeError):
    pass


def cost(input_tokens, output_tokens):
    return (
        math.ceil(input_tokens * INPUT_PER_MILLION + output_tokens * OUTPUT_PER_MILLION) / 1_000_000
    )


class CostGuard:
    def __init__(self, path: Path, limit_usd=10.0):
        if not 0 < limit_usd <= 10:
            raise ValueError("This run is approved for at most $10")
        self.path = path
        self.limit = limit_usd
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self):
        if self.path.exists():
            ledger = json.loads(self.path.read_text())
            if ledger["limit_usd"] != self.limit:
                raise ValueError("Cannot change the budget of an existing run")
            return ledger
        return {
            "limit_usd": self.limit,
            "pricing_source": PRICING_SOURCE,
            "pricing_checked": "2026-09-08",
            "input_rate": INPUT_PER_MILLION,
            "output_rate": OUTPUT_PER_MILLION,
            "requests": [],
            "accounted_usd": 0.0,
            "scope": "This evaluation only; excludes taxes and other AWS resources",
        }

    def request(self, delegate, prompt, max_output_tokens):
        if not isinstance(max_output_tokens, int) or not 1 <= max_output_tokens <= 6000:
            raise ValueError("Output exceeds approved per-request token bound")
        # Byte count deliberately overestimates text BPE tokens; headroom covers
        # the Responses envelope. Large contexts are rejected, not repriced.
        input_bound = len(json.dumps(prompt, separators=(",", ":")).encode()) + 1024
        if input_bound > 200_000:
            raise ValueError("Request exceeds the short-context budget guard")
        reserve = cost(input_bound, max_output_tokens)
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ledger = self.load()
            spent = sum(r["accounted_usd"] for r in ledger["requests"])
            if spent + reserve > self.limit or len(ledger["requests"]) >= 96:
                raise BudgetExceeded("Next request could exceed the $10 or 96-attempt limit")
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
