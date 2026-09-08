from evaluation.intelligence_benchmark import run


def test_j2_intelligence_benchmark_passes() -> None:
    report = run()

    assert report["passed"] is True
    assert report["metrics"]["fixture_documents"] >= 120
    assert report["metrics"]["acl_leakage_count"] == 0
