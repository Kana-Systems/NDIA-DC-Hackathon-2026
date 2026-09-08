"""Inference for locally trained CUAD weights; no executable pickle artifacts."""

import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer

from ml.windowing import sliding_text_windows


class LinearClassifier:
    def __init__(self, path: Path):
        self.config = json.loads((path / "linear_config.json").read_text())
        weights = np.load(path / "linear_weights.npz", allow_pickle=False)
        self.vectorizer = TfidfVectorizer(
            vocabulary=self.config["vocabulary"],
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.vectorizer.idf_ = weights["idf"]
        self.coefficients = weights["coefficients"]
        self.intercepts = weights["intercepts"]
        self.model_id = self.config["model_id"]

    def predict(self, text: str, threshold: float = 0.5) -> dict:
        windowing = self.config["windowing"]
        windows = sliding_text_windows(text, windowing["window_chars"], windowing["stride_chars"])
        matrix = self.vectorizer.transform(windows)
        scores = expit(matrix @ self.coefficients.T + self.intercepts).max(axis=0)
        return {
            "model_id": self.model_id,
            "labels": [
                {"label": label, "score": float(score)}
                for label, score in zip(self.config["labels"], scores, strict=True)
                if score >= threshold
            ],
        }
