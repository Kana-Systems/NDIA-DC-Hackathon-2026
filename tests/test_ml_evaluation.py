import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from evaluation.benchmark import analyze, run
from ml.heuristic import HeuristicClassifier
from ml.inference import TransformerClassifier, input_fn, output_fn, predict_fn
from ml.launch_sagemaker import estimator_kwargs, validate_config
from ml.metrics import multilabel_metrics
from ml.preprocess_cuad import cuad_category, examples, labels_for_bounds, split
from ml.train import (
    apply_window_config,
    build_parser,
    load_training_contract,
    load_window_config,
    package_artifacts,
    training_api_kwargs,
    tuned_micro_f1,
)

ROOT = Path(__file__).resolve().parents[1]


class MlEvaluationTests(unittest.TestCase):
    def test_training_and_full_contract_windows_share_overlap_label_rule(self):
        spans = [{"label": "termination", "start": 80, "end": 120}]

        self.assertEqual(labels_for_bounds(spans, 0, 100), ["termination"])
        self.assertEqual(labels_for_bounds(spans, 0, 99), [])

    def test_tuned_micro_f1_finds_exact_global_threshold(self):
        result = tuned_micro_f1(
            [[2.1972246, 1.3862944, 0.8472979, -2.1972246]],
            [[1, 0, 1, 0]],
        )

        self.assertAlmostEqual(result["micro_f1_tuned"], 0.8)
        self.assertAlmostEqual(result["micro_f1_tuned_threshold"], 0.7)
        self.assertAlmostEqual(result["micro_f1_tuned_precision"], 2 / 3)
        self.assertEqual(result["micro_f1_tuned_recall"], 1.0)

    def test_cuad_preprocessing_and_document_split(self):
        fixture = ROOT / "tests" / "fixtures" / "ml" / "cuad_realistic.json"
        dataset = json.loads(fixture.read_text(encoding="utf-8"))
        items = examples(dataset)
        self.assertEqual(
            items[0]["labels"],
            [
                "ip_ownership_assignment",
                "termination_for_convenience",
                "uncapped_liability",
            ],
        )
        self.assertNotIn("limeenergyco_09_09_1999", items[0]["labels"])
        self.assertEqual(
            cuad_category(
                "ignored",
                "DOCUMENT-NAME__Termination For Convenience_12",
            ),
            "termination_for_convenience",
        )
        train, validation = split(items, 20)
        self.assertTrue(not train or not validation)
        self.assertEqual(len(train) + len(validation), 1)

    def test_answer_centered_windows_keep_late_spans_and_negatives(self):
        prefix = "Neutral contract language. " * 100
        termination = "Either party may terminate on thirty days notice."
        ownership = "All new intellectual property is assigned to Supplier."
        context = f"{prefix}{termination} Nearby clause. {ownership}"
        termination_start = context.index(termination)
        ownership_start = context.index(ownership)
        dataset = {
            "data": [
                {
                    "title": "LATE-CONTRACT",
                    "paragraphs": [
                        {
                            "context": context,
                            "qas": [
                                {
                                    "id": "LATE-CONTRACT__Termination For Convenience_0",
                                    "question": '"Termination For Convenience"',
                                    "answers": [
                                        {
                                            "text": termination,
                                            "answer_start": termination_start,
                                        }
                                    ],
                                },
                                {
                                    "id": "LATE-CONTRACT__IP Ownership Assignment_1",
                                    "question": '"IP Ownership Assignment"',
                                    "answers": [
                                        {
                                            "text": ownership,
                                            "answer_start": ownership_start,
                                        }
                                    ],
                                },
                            ],
                        }
                    ],
                }
            ]
        }
        items = examples(dataset, window_chars=240, stride_chars=120)
        positives = [item for item in items if not item["is_negative"]]
        negatives = [item for item in items if item["is_negative"]]
        self.assertGreater(termination_start, 2000)
        self.assertTrue(any(termination in item["text"] for item in positives))
        self.assertTrue(any(ownership in item["text"] for item in positives))
        self.assertTrue(
            any(
                set(item["labels"]) == {"termination_for_convenience", "ip_ownership_assignment"}
                for item in positives
            )
        )
        self.assertTrue(negatives)
        self.assertTrue(all(len(item["text"]) <= 240 for item in items))
        self.assertTrue(all(termination not in item["text"] for item in negatives))
        train, validation = split(items, 20)
        self.assertFalse(
            {item["document_id"] for item in train} & {item["document_id"] for item in validation}
        )

    def test_cuad_domain_mapping_is_explicit_and_non_applicability(self):
        path = ROOT / "ml" / "cuad_category_domain_mapping.json"
        mapping = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn(
            "cannot establish FAR or DFARS applicability", mapping["applicability_warning"]
        )
        self.assertIn(
            "termination", mapping["mappings"]["termination_for_convenience"]["domain_labels"]
        )
        self.assertIn(
            "data_rights", mapping["mappings"]["ip_ownership_assignment"]["domain_labels"]
        )
        self.assertIn(
            "confidentiality_cyber_adjacent",
            mapping["mappings"]["uncapped_liability"]["domain_labels"],
        )

    def test_heuristic_inference_contract(self):
        payload = input_fn('{"text":"NIST SP 800-171 applies."}', "application/json")
        result = predict_fn(payload, HeuristicClassifier())
        self.assertEqual(result["predictions"][0]["labels"][0]["label"], "cybersecurity")
        body, content_type = output_fn(result, "application/json")
        self.assertIn("predictions", body)
        self.assertEqual(content_type, "application/json")

    def test_transformer_inference_windows_and_aggregates_per_input(self):
        class FakePredictor:
            def __init__(self):
                self.windows = []

            def __call__(self, windows, **_kwargs):
                self.windows.extend(windows)
                return [
                    [
                        {
                            "label": "termination_for_convenience",
                            "score": 0.9 if "LATE" in window else 0.2,
                        },
                        {
                            "label": "ip_ownership_assignment",
                            "score": 0.8 if "IPMARK" in window else 0.1,
                        },
                    ]
                    for window in windows
                ]

        predictor = FakePredictor()
        model = TransformerClassifier(
            predictor,
            window_chars=40,
            stride_chars=20,
            tokenizer_max_length=32,
        )
        payload = {
            "texts": ["early " * 20 + "IPMARK middle " + "late " * 20 + "LATE", "short"],
            "threshold": 0.5,
        }
        result = predict_fn(payload, model)
        self.assertGreater(len(predictor.windows), 2)
        self.assertEqual(len(result["predictions"]), 2)
        labels = {item["label"]: item["score"] for item in result["predictions"][0]["labels"]}
        self.assertEqual(labels["termination_for_convenience"], 0.9)
        self.assertEqual(labels["ip_ownership_assignment"], 0.8)
        self.assertEqual(result["predictions"][1]["labels"], [])

    def test_metrics(self):
        metrics = multilabel_metrics([["a"], ["b"]], [["a"], ["c"]])
        self.assertEqual(metrics["micro_f1"], 0.5)
        self.assertEqual(metrics["exact_match"], 0.5)

    def test_policy_analysis(self):
        metadata = {
            "agency_department": "DoD",
            "instrument_type": "contract",
            "acquisition_solely_cots": False,
        }
        findings = analyze("FAR 52.204-7 applies.", metadata)
        self.assertEqual(
            [item["id"] for item in findings], ["missing-required-clause:252.204-7012"]
        )
        cots = {**metadata, "acquisition_solely_cots": True}
        self.assertEqual(analyze("FAR 52.204-7 applies.", cots), [])
        civilian = {**metadata, "agency_department": "Department of Homeland Security"}
        self.assertEqual(analyze("FAR 52.204-7 applies.", civilian), [])
        self.assertEqual(
            analyze("Covered defense information. FAR 52.204-7 applies.", {}),
            [],
        )

    def test_training_contract_and_artifact_packaging_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = ["termination_for_convenience", "ip_ownership_assignment"]
            (root / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
            for split_name, label in (
                ("train", None),
                ("validation", labels[1]),
            ):
                record = {
                    "id": split_name,
                    "text": "synthetic",
                    "labels": [label] if label else [],
                }
                (root / f"{split_name}.jsonl").write_text(
                    json.dumps(record) + "\n",
                    encoding="utf-8",
                )
            window_config = {
                "schema_version": "1.0",
                "strategy": "answer_centered_positive_sliding_negative",
                "window_chars": 1800,
                "stride_chars": 900,
                "negative_windows_per_positive": 1,
            }
            (root / "window_config.json").write_text(
                json.dumps(window_config),
                encoding="utf-8",
            )
            loaded_window_config = load_window_config(root)
            loaded_labels, train, validation = load_training_contract(
                root,
                "multi_label",
                loaded_window_config["window_chars"],
            )
            self.assertEqual(loaded_labels, labels)
            self.assertEqual(train[0]["labels"], [])
            self.assertEqual(len(train), 1)
            self.assertEqual(len(validation), 1)

            args = build_parser().parse_args(
                [
                    "--model-name",
                    "local/legal-model",
                    "--training-dir",
                    str(root),
                    "--model-dir",
                    str(root / "model"),
                ]
            )
            package_artifacts(
                root / "model",
                labels,
                args,
                {"eval_micro_f1": 0.5},
                loaded_window_config,
            )
            label_mapping = json.loads(
                (root / "model" / "label_mapping.json").read_text(encoding="utf-8")
            )
            self.assertEqual(label_mapping["id2label"]["0"], labels[0])
            self.assertTrue((root / "model" / "training_provenance.json").exists())
            self.assertTrue((root / "model" / "code" / "inference.py").exists())
            self.assertTrue((root / "model" / "code" / "windowing.py").exists())
            self.assertTrue((root / "model" / "CUAD_ATTRIBUTION.md").exists())
            provenance = json.loads(
                (root / "model" / "training_provenance.json").read_text(encoding="utf-8")
            )
            self.assertEqual(provenance["windowing"]["window_chars"], 1800)
            model_config = SimpleNamespace()
            apply_window_config(model_config, loaded_window_config, 512)
            self.assertEqual(model_config.contract_window_chars, 1800)
            self.assertEqual(model_config.contract_window_stride_chars, 900)
            self.assertEqual(model_config.contract_tokenizer_max_length, 512)

    def test_current_transformers_and_sagemaker_configuration_contracts(self):
        class CurrentTrainingArguments:
            def __init__(self, output_dir=None, eval_strategy=None):
                pass

        args = build_parser().parse_args([])
        training_kwargs = training_api_kwargs(CurrentTrainingArguments, args)
        self.assertEqual(training_kwargs["eval_strategy"], "epoch")
        self.assertEqual(training_kwargs["metric_for_best_model"], "micro_f1")
        self.assertNotIn("evaluation_strategy", training_kwargs)
        recall_args = build_parser().parse_args(["--metric-for-best-model", "micro_f2"])
        recall_kwargs = training_api_kwargs(CurrentTrainingArguments, recall_args)
        self.assertEqual(recall_kwargs["metric_for_best_model"], "micro_f2")

        config = validate_config(
            {
                "instance_type": "ml.g6.xlarge",
                "transformers_version": "4.36",
                "pytorch_version": "2.1",
                "py_version": "py310",
                "tags": [{"Key": "Project", "Value": "contract-review"}],
                "hyperparameters": {"problem-type": "multi_label"},
            }
        )
        kwargs = estimator_kwargs(config, "arn:aws-us-gov:iam::123:role/test")
        self.assertTrue(Path(kwargs["source_dir"]).is_absolute())
        self.assertEqual(kwargs["entry_point"], "train.py")
        self.assertEqual(kwargs["instance_type"], "ml.g6.xlarge")
        self.assertEqual(kwargs["tags"], [{"Key": "Project", "Value": "contract-review"}])

    def test_offline_benchmark_matches_expected_outputs(self):
        report = run()
        self.assertTrue(report["passed"])
        self.assertEqual(report["metrics"]["exact_match"], 1.0)


if __name__ == "__main__":
    unittest.main()
