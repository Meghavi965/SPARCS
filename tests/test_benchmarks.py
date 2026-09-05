from pathlib import Path


def test_benchmark_dataset_files_exist():
    root = Path(__file__).resolve().parent.parent
    for rel in [
        "benchmarks/datasets/harmbench_subset.json",
        "benchmarks/datasets/advglue_rag_subset.json",
        "benchmarks/datasets/gcg_suffixes.json",
    ]:
        assert (root / rel).exists(), f"missing dataset: {rel}"


def test_run_eval_exports_report_data():
    from benchmarks.run_eval import evaluate_suite

    report = evaluate_suite()
    assert isinstance(report, dict)
    assert "metrics" in report
    assert "table_1" in report["metrics"]
    assert "table_2" in report["metrics"]
