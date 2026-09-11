from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .types import Song
from .audio import FEATURE_NAMES, feature_vector


_SCHEMA_VERSION = 2
_EMPTY_TEXT_SENTINEL = "[no_text]"
_MODEL_METADATA_REPEATS = 5
_SCORE_MODES = frozenset({"probability", "softmax"})


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
    """Compose the model input with metadata repeated for a strong prior.

    The official workbook stores title and artist at the front of ``text``.
    Strip that duplicated prefix before adding the metadata block so a song
    does not get an accidental tenfold title weight.
    """
    raw_text = _text_or_sentinel(song.text)
    if raw_text == _EMPTY_TEXT_SENTINEL:
        lyric = ""
    else:
        prefix = " ".join(
            part.strip() for part in (song.name, song.artists) if str(part).strip()
        )
        lyric = raw_text
        if prefix and raw_text.startswith(prefix):
            lyric = raw_text[len(prefix) :].strip()
    metadata = " ".join(
        part.strip()
        for part in (song.name, song.artists, song.genre)
        if str(part).strip()
    )
    composed = " ".join(
        part for part in (" ".join([metadata] * _MODEL_METADATA_REPEATS), lyric) if part
    ).strip()
    return composed or _text_or_sentinel(song.name)


def compose_model_text(song: Song) -> str:
    """Return the exact text representation used by training and inference."""
    return _song_text(song)


def _label_order_sha256(labels: tuple[str, ...]) -> str:
    canonical = json.dumps(
        list(labels), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass
class TextScorer:
    labels: tuple[str, ...] = ()
    vectorizer: TfidfVectorizer | None = None
    classifier: OneVsRestClassifier | None = None
    score_mode: str = "probability"

    def fit(self, songs: list[Song], labels: tuple[str, ...]) -> TextScorer:
        configured_labels = _validate_labels(labels)
        if not songs:
            raise ValueError("songs must be nonempty")
        configured_label_set = set(configured_labels)
        unknown_labels = {
            label for song in songs for label in song.labels if label not in configured_label_set
        }
        if unknown_labels:
            raise ValueError(
                "songs contain labels outside the configuration: "
                + ", ".join(sorted(map(str, unknown_labels)))
            )
        if len(configured_labels) == 1:
            raise ValueError(
                "a one-label configuration cannot provide real negatives"
            )

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
            ngram_range=(1, 5),
            min_df=1,
            max_features=150000,
            sublinear_tf=True,
        )
        features = vectorizer.fit_transform([_song_text(song) for song in songs])
        classifier = OneVsRestClassifier(
            LinearSVC(C=0.3, class_weight="balanced")
        )
        classifier.fit(features, targets)

        self.labels = configured_labels
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.score_mode = "softmax"
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
        return self._classifier_scores(classifier, features, len(texts))

    def _fitted_components(self) -> tuple[TfidfVectorizer, OneVsRestClassifier]:
        _validate_labels(self.labels)
        if self.vectorizer is None or self.classifier is None:
            raise ValueError("TextScorer has not been fitted")
        try:
            check_is_fitted(self.vectorizer)
            check_is_fitted(self.classifier)
            if len(self.classifier.estimators_) != len(self.labels):
                raise ValueError("classifier estimators do not match configured labels")
            probe = self.vectorizer.transform([_EMPTY_TEXT_SENTINEL])
            if self.score_mode not in _SCORE_MODES:
                raise ValueError("unsupported score mode")
            self._classifier_scores(self.classifier, probe, 1)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("TextScorer has not been fitted") from error
        return self.vectorizer, self.classifier

    def _classifier_scores(
        self,
        classifier: OneVsRestClassifier,
        features: Any,
        rows: int,
    ) -> np.ndarray:
        if self.score_mode == "softmax":
            margins = np.asarray(classifier.decision_function(features), dtype=float)
            if margins.ndim == 1:
                margins = margins.reshape(-1, 1)
            if margins.shape != (rows, len(self.labels)):
                raise ValueError("classifier scores do not match configured labels")
            if not np.isfinite(margins).all():
                raise ValueError("classifier returned invalid scores")
            shifted = margins - np.max(margins, axis=1, keepdims=True)
            probabilities = np.exp(np.clip(shifted, -80.0, 0.0))
            probabilities /= probabilities.sum(axis=1, keepdims=True)
        else:
            probabilities = np.asarray(classifier.predict_proba(features), dtype=float)
        if len(self.labels) == 1:
            classes = np.asarray(getattr(classifier, "classes_", ()))
            positive_indices = np.flatnonzero(classes == 1)
            if probabilities.shape != (rows, len(classes)) or len(positive_indices) != 1:
                raise ValueError("classifier probabilities do not match configured labels")
            probabilities = probabilities[:, positive_indices]
        if probabilities.shape != (rows, len(self.labels)):
            raise ValueError("classifier probabilities do not match configured labels")
        if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError("classifier returned invalid probabilities")
        return probabilities


@dataclass
class AudioScorer:
    labels: tuple[str, ...] = ()
    scaler: StandardScaler | None = None
    classifier: OneVsRestClassifier | None = None

    def fit(
        self,
        samples: list[tuple[Song, dict[str, float]]],
        labels: tuple[str, ...],
    ) -> AudioScorer:
        configured_labels = _validate_labels(labels)
        if not samples:
            raise ValueError("samples must be nonempty")
        configured_label_set = set(configured_labels)
        unknown_labels = {
            label for song, _ in samples for label in song.labels if label not in configured_label_set
        }
        if unknown_labels:
            raise ValueError(
                "songs contain labels outside the configuration: "
                + ", ".join(sorted(map(str, unknown_labels)))
            )
        if len(configured_labels) == 1:
            raise ValueError("a one-label configuration cannot provide real negatives")

        song_ids = [song.song_id for song, _ in samples]
        if len(song_ids) != len(set(song_ids)):
            raise ValueError("duplicated song ID in audio samples")
        matrix = np.vstack([self._validated_feature_vector(features) for _, features in samples])

        encoder = MultiLabelBinarizer(classes=configured_labels)
        targets = encoder.fit_transform([song.labels for song, _ in samples])
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

        scaler = StandardScaler()
        normalized = scaler.fit_transform(matrix)
        classifier = OneVsRestClassifier(
            LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
        )
        classifier.fit(normalized, targets)
        self.labels = configured_labels
        self.scaler = scaler
        self.classifier = classifier
        return self

    def score(self, features: dict[str, float]) -> dict[str, float]:
        probabilities = self.score_many([features])
        return {
            label: float(probability)
            for label, probability in zip(self.labels, probabilities[0], strict=True)
        }

    def score_many(self, features: list[dict[str, float]]) -> np.ndarray:
        scaler, classifier = self._fitted_components()
        if not features:
            return np.empty((0, len(self.labels)), dtype=float)
        matrix = np.vstack([self._validated_feature_vector(record) for record in features])
        probabilities = np.asarray(classifier.predict_proba(scaler.transform(matrix)), dtype=float)
        if probabilities.shape != (len(features), len(self.labels)):
            raise ValueError("classifier probabilities do not match configured labels")
        if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError("classifier returned invalid probabilities")
        return probabilities

    def _fitted_components(self) -> tuple[StandardScaler, OneVsRestClassifier]:
        _validate_labels(self.labels)
        if self.scaler is None or self.classifier is None:
            raise ValueError("AudioScorer has not been fitted")
        try:
            check_is_fitted(self.scaler)
            check_is_fitted(self.classifier)
            if len(self.classifier.estimators_) != len(self.labels):
                raise ValueError("classifier estimators do not match configured labels")
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("AudioScorer has not been fitted") from error
        return self.scaler, self.classifier

    @staticmethod
    def _validated_feature_vector(features: dict[str, float]) -> np.ndarray:
        vector = feature_vector(features)
        bounds = {
            "rms_db": (-300.0, 20.0),
            "zero_crossing_rate": (0.0, 1.0),
            "spectral_centroid_hz": (0.0, 24_000.0),
            "dynamic_range_db": (0.0, 300.0),
        }
        for value, name in zip(vector, FEATURE_NAMES, strict=True):
            lower, upper = bounds[name]
            if not lower <= value <= upper:
                raise ValueError(f"feature {name} is out of range")
        return vector


def save_text_scorer(scorer: TextScorer, path: str | Path) -> None:
    vectorizer, classifier = scorer._fitted_components()
    trained_labels = tuple(scorer.labels)
    joblib.dump(
        {
            "schema_version": _SCHEMA_VERSION,
            "labels": trained_labels,
            "trained_labels": trained_labels,
            "label_order_sha256": _label_order_sha256(trained_labels),
            "vectorizer": vectorizer,
            "classifier": classifier,
            "score_mode": scorer.score_mode,
        },
        Path(path),
    )


def load_text_scorer(path: str | Path | Any, *, trusted: bool = False) -> TextScorer:
    """Load a scorer only after the caller explicitly trusts its joblib artifact.

    Joblib deserializes pickle data, so callers must set ``trusted=True`` only
    for artifacts from a source they trust.
    """
    if not trusted:
        raise ValueError("joblib artifacts must be trusted; pass trusted=True")
    try:
        # A service can pass an already-validated open file handle so the
        # artifact cannot be swapped between path validation and deserialization.
        source = path if hasattr(path, "read") else Path(path)
        record: Any = joblib.load(source)
    except Exception as error:
        raise ValueError("invalid text scorer payload") from error
    if not isinstance(record, dict):
        raise ValueError("unsupported text scorer schema")
    if record.get("schema_version") == 1:
        raise ValueError(
            "schema version 1 is incompatible with bound model artifacts; retrain the scorer"
        )
    if record.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unsupported text scorer schema")
    required = {
        "labels",
        "trained_labels",
        "label_order_sha256",
        "vectorizer",
        "classifier",
    }
    if not required.issubset(record):
        raise ValueError("text scorer payload is missing required components")
    raw_labels = record["labels"]
    raw_trained_labels = record["trained_labels"]
    if not isinstance(raw_labels, tuple) or not isinstance(raw_trained_labels, tuple):
        raise ValueError("text scorer payload has invalid labels")
    try:
        labels = _validate_labels(raw_labels)
        trained_labels = _validate_labels(raw_trained_labels)
    except (TypeError, ValueError) as error:
        raise ValueError("text scorer payload has invalid labels") from error
    if labels != trained_labels:
        raise ValueError("payload labels must exactly match trained_labels")
    label_order_sha256 = record["label_order_sha256"]
    if (
        not isinstance(label_order_sha256, str)
        or label_order_sha256 != _label_order_sha256(trained_labels)
    ):
        raise ValueError("label_order_sha256 does not match trained_labels")
    vectorizer = record["vectorizer"]
    classifier = record["classifier"]
    if not isinstance(vectorizer, TfidfVectorizer) or not isinstance(classifier, OneVsRestClassifier):
        raise ValueError("text scorer payload has invalid components")
    score_mode = record.get("score_mode", "probability")
    if not isinstance(score_mode, str) or score_mode not in _SCORE_MODES:
        raise ValueError("text scorer payload has invalid score mode")
    scorer = TextScorer(
        labels=labels,
        vectorizer=vectorizer,
        classifier=classifier,
        score_mode=score_mode,
    )
    try:
        scorer._fitted_components()
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("text scorer payload has invalid components") from error
    return scorer
