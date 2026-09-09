"""Load a packaged classifier and exercise its SageMaker inference contract."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON_CONTENT_TYPE = "application/json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.inference import input_fn, model_fn, output_fn, predict_fn  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--expected-model-id", required=True)
    parser.add_argument("--minimum-label-count", type=int, default=1)
    args = parser.parse_args()

    model = model_fn(str(args.model_dir.resolve()))
    payload = input_fn(
        json.dumps(
            {
                "texts": [
                    "Either party may terminate this agreement for convenience "
                    "upon thirty days written notice."
                ],
                "threshold": 0.0,
            }
        ),
        JSON_CONTENT_TYPE,
    )
    result = predict_fn(payload, model)
    encoded, content_type = output_fn(result, JSON_CONTENT_TYPE)
    decoded = json.loads(encoded)
    predictions = decoded.get("predictions", [])
    if content_type != JSON_CONTENT_TYPE or len(predictions) != 1:
        raise RuntimeError("model returned an invalid SageMaker response")
    prediction = predictions[0]
    if prediction.get("model_id") != args.expected_model_id:
        raise RuntimeError("model returned an unexpected model ID")
    labels = prediction.get("labels")
    if not isinstance(labels, list) or len(labels) < args.minimum_label_count:
        raise RuntimeError("model returned too few labels at a zero threshold")
    names = [item.get("label") for item in labels]
    scores = [item.get("score") for item in labels]
    if (
        any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
        or any(
            not isinstance(score, (float, int))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
            or not 0 <= float(score) <= 1
            for score in scores
        )
    ):
        raise RuntimeError("model returned invalid labels")
    leaders = sorted(labels, key=lambda item: item["score"], reverse=True)[:3]
    print(
        json.dumps(
            {
                "content_type": content_type,
                "label_count": len(labels),
                "model_id": prediction["model_id"],
                "top_labels": leaders,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
