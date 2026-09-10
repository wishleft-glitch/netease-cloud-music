from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.validation import check_is_fitted

from .types import Song


_SCHEMA_VERSION = 1
_EMPTY_TEXT_SENTINEL = "[no_text]"


def _validate_labels(labels: tuple[str, ...]) -> tuple[str, ...]:
    if not labels:
        raise ValueError("labels must be nonempty")
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("labels must be nonblank strings")
    if len(set(labels)) != len(labels):
        raise ValueError("labels must be unique")
    return tuple(labels)


def _text_or_sentinel(value: object) -> str:
    if value is None:
        return _EMPTY_TEXT_SENTINEL
    text = str(value).strip()
    return text or _EMPTY_TEXT_SENTINEL


def _song_text(song: Song) -> str:
    text = _text_or_sentinel(song.text)
    return text if text != _EMPTY_TEXT_SENTINEL else _text_or_sentinel(song.name)


@dataclass
class TextScorer:
    labels: tuple[str, ...] = ()
    vectorizer: TfidfVectorizer | None = None
    classifier: OneVsRestClassifier | None = None

    def fit(self, songs: list[Song], labels: tuple[str, ...]) -> TextScorer:
        configured_labels = _validate_labels(labels)
        if not songs:
            raise ValueError("songs must be nonempty")

        encoder = MultiLabelBinarizer(classes=configured_labels)
        targets = encoder.fit_transform([song.labels for song in songs])
        unsupported = [
            label
            for index, label in enumerate(configured_labels)
            if not targets[:, index].any() or targets[:, index].all()
        ]
        if unsupported:
            raise ValueError(
                "labels require both positive and negative training examples: "
                + ", ".join(unsupported)
            )

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
            max_features=80000,
            sublinear_tf=True,
        )
        features = vectorizer.fit_transform([_song_text(song) for song in songs])
        classifier = OneVsRestClassifier(
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="liblinear",
            )
        )
        classifier.fit(features, targets)

        self.labels = configured_labels
        self.vectorizer = vectorizer
        self.classifier = classifier
        return self

    def score(self, text: str) -> dict[str, float]:
        probabilities = self.score_many([text])
        return {
            label: float(probability)
            for label, probability in zip(self.labels, probabilities[0], strict=True)
        }

    def score_many(self, texts: list[str]) -> np.ndarray:
        vectorizer, classifier = self._fitted_components()
        if not texts:
            return np.empty((0, len(self.labels)), dtype=float)
        features = vectorizer.transform([_text_or_sentinel(text) for text in texts])
        probabilities = np.asarray(classifier.predict_proba(features), dtype=float)
        if probabilities.shape != (len(texts), len(self.labels)):
            raise ValueError("classifier probabilities do not match configured labels")
        if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError("classifier returned invalid probabilities")
        return probabilities

    def _fitted_components(self) -> tuple[TfidfVectorizer, OneVsRestClassifier]:
        _validate_labels(self.labels)
        if self.vectorizer is None or self.classifier is None:
            raise ValueError("TextScorer has not been fitted")
        try:
            check_is_fitted(self.vectorizer)
            check_is_fitted(self.classifier)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("TextScorer has not been fitted") from error
        return self.vectorizer, self.classifier


def save_text_scorer(scorer: TextScorer, path: str | Path) -> None:
    vectorizer, classifier = scorer._fitted_components()
    joblib.dump(
        {
            "schema_version": _SCHEMA_VERSION,
            "labels": scorer.labels,
            "vectorizer": vectorizer,
            "classifier": classifier,
        },
        Path(path),
    )


def load_text_scorer(path: str | Path) -> TextScorer:
    try:
        record: Any = joblib.load(Path(path))
    except Exception as error:
        raise ValueError("invalid text scorer payload") from error
    if not isinstance(record, dict) or record.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unsupported text scorer schema")
    required = {"labels", "vectorizer", "classifier"}
    if not required.issubset(record):
        raise ValueError("text scorer payload is missing required components")
    raw_labels = record["labels"]
    if not isinstance(raw_labels, tuple):
        raise ValueError("text scorer payload has invalid labels")
    try:
        labels = _validate_labels(raw_labels)
    except (TypeError, ValueError) as error:
        raise ValueError("text scorer payload has invalid labels") from error
    vectorizer = record["vectorizer"]
    classifier = record["classifier"]
    if not isinstance(vectorizer, TfidfVectorizer) or not isinstance(classifier, OneVsRestClassifier):
        raise ValueError("text scorer payload has invalid components")
    scorer = TextScorer(labels=labels, vectorizer=vectorizer, classifier=classifier)
    try:
        scorer._fitted_components()
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("text scorer payload has invalid components") from error
    return scorer
