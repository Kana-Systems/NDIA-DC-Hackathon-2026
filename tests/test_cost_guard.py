from types import SimpleNamespace

import pytest

from evaluation.cost_guard import BudgetedAdapter, BudgetExceeded, CostGuard, cost


class FakeAdapter:
    settings = SimpleNamespace(bedrock_model_id="openai.gpt-5.6-terra", aws_region="us-gov-west-1")

    def __init__(self, guard, *, fail=False, usage=True):
        self.guard, self.fail, self.usage = guard, fail, usage
        self.calls = 0

    def request_json(self, prompt, **kwargs):
        self.calls += 1
        assert self.guard.load()["requests"][-1]["state"] == "reserved"
        if self.fail:
            raise TimeoutError("Simulated interrupted connection")
        self.last_usage = {"input_tokens": 10, "output_tokens": 5} if self.usage else {}
        return {"ok": True}


def test_reserve_before_call_and_account_actual_usage(tmp_path):
    guard = CostGuard(tmp_path / "cost.json")
    adapter = FakeAdapter(guard)
    assert BudgetedAdapter(adapter, guard).request_json({"text": "hello"}, 100) == {"ok": True}
    ledger = guard.load()
    assert ledger["accounted_usd"] == cost(10, 5)
    assert ledger["requests"][0]["reserved_usd"] > ledger["accounted_usd"]


def test_no_request_when_allowance_would_exceed_limit(tmp_path):
    guard = CostGuard(tmp_path / "cost.json", 0.001)
    adapter = FakeAdapter(guard)
    with pytest.raises(BudgetExceeded):
        guard.request(adapter, {"text": "hello"}, 100)
    assert adapter.calls == 0


@pytest.mark.parametrize("fail", [True, False])
def test_failures_and_unknown_usage_keep_reservation_across_restart(tmp_path, fail):
    path = tmp_path / "cost.json"
    guard = CostGuard(path)
    adapter = FakeAdapter(guard, fail=fail, usage=False)
    if fail:
        with pytest.raises(TimeoutError):
            guard.request(adapter, {}, 100)
    else:
        guard.request(adapter, {}, 100)
    ledger = CostGuard(path).load()
    assert ledger["accounted_usd"] == ledger["requests"][0]["reserved_usd"]


def test_wrong_model_or_oversized_request_rejected(tmp_path):
    guard = CostGuard(tmp_path / "cost.json")
    with pytest.raises(ValueError):
        guard.request(FakeAdapter(guard), {"text": "a" * 200000}, 100)
    adapter = FakeAdapter(guard)
    adapter.settings = SimpleNamespace(bedrock_model_id="different", aws_region="us-gov-west-1")
    with pytest.raises(ValueError):
        BudgetedAdapter(adapter, guard)
    with pytest.raises(ValueError):
        CostGuard(tmp_path / "other.json", 11)
