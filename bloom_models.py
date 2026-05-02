from __future__ import annotations

import re
from typing import Dict, List, Sequence

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize

from encoder_backends import StableTextEncoder


BLOOM_LEVELS: List[str] = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create",
]

LOWER_ORDER = ["Remember", "Understand", "Apply"]
HIGHER_ORDER = ["Analyze", "Evaluate", "Create"]

BLOOM_ACTION_VERBS: Dict[str, List[str]] = {
    "Remember": [
        "define", "describe", "identify", "label", "list", "match", "name",
        "recall", "recognize", "select", "state",
    ],
    "Understand": [
        "classify", "compare", "contrast", "discuss", "explain", "illustrate",
        "interpret", "outline", "paraphrase", "summarize",
    ],
    "Apply": [
        "apply", "calculate", "demonstrate", "determine", "execute",
        "implement", "solve", "use",
    ],
    "Analyze": [
        "analyze", "differentiate", "distinguish", "examine", "infer",
        "investigate", "organize", "relate",
    ],
    "Evaluate": [
        "appraise", "argue", "assess", "critique", "defend", "evaluate",
        "judge", "justify", "recommend",
    ],
    "Create": [
        "compose", "construct", "create", "design", "develop", "formulate",
        "generate", "plan", "produce", "propose", "synthesize",
    ],
}

_QUESTION_STARTERS = {"what", "why", "how", "when", "where", "which", "who"}


class BloomCueTransformer(BaseEstimator, TransformerMixin):
    """Domain-light Bloom cue features focused on verbs and question form.

    TF-IDF baselines tend to learn subject vocabulary. These features provide a
    compact domain-invariant channel that emphasizes cognitive demand cues.
    """

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = bool(normalize)
        self.feature_names_: List[str] = []

    def fit(self, X: Sequence[str], y: Sequence[str] | None = None):
        del X, y
        names: List[str] = []
        for level in BLOOM_LEVELS:
            names.append(f"cue_count_{level.lower()}")
        for level in BLOOM_LEVELS:
            names.append(f"cue_present_{level.lower()}")
        names.extend(
            [
                "starts_with_question_word",
                "has_question_mark",
                "token_count_log",
                "first_token_is_bloom_verb",
            ]
        )
        self.feature_names_ = names
        return self

    def transform(self, X: Sequence[str]):
        rows = []
        for text in X:
            raw = str(text).lower()
            tokens = re.findall(r"[a-z]+", raw)
            token_count = max(1, len(tokens))
            token_set = set(tokens)

            counts = []
            present = []
            for level in BLOOM_LEVELS:
                verbs = BLOOM_ACTION_VERBS[level]
                count = sum(1 for token in tokens if token in verbs)
                value = count / token_count if self.normalize else float(count)
                counts.append(float(value))
                present.append(1.0 if token_set.intersection(verbs) else 0.0)

            first = tokens[0] if tokens else ""
            all_verbs = {verb for verbs in BLOOM_ACTION_VERBS.values() for verb in verbs}
            rows.append(
                counts
                + present
                + [
                    1.0 if first in _QUESTION_STARTERS else 0.0,
                    1.0 if "?" in raw else 0.0,
                    float(np.log1p(token_count)),
                    1.0 if first in all_verbs else 0.0,
                ]
            )
        return sparse.csr_matrix(np.asarray(rows, dtype=np.float32))


def make_feature_union() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=30000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def make_domain_robust_feature_union() -> FeatureUnion:
    return FeatureUnion(
        [
            ("lexical", make_feature_union()),
            ("bloom_cues", BloomCueTransformer(normalize=True)),
        ]
    )


def make_bloom_cue_logreg_pipeline(class_weight: str | None = "balanced") -> Pipeline:
    return Pipeline(
        [
            ("features", BloomCueTransformer(normalize=True)),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    class_weight=class_weight,
                ),
            ),
        ]
    )


def make_linear_svm_pipeline(class_weight: str | None = "balanced") -> Pipeline:
    return Pipeline(
        [
            ("features", make_feature_union()),
            ("clf", LinearSVC(class_weight=class_weight)),
        ]
    )


def make_domain_robust_linear_svm_pipeline(class_weight: str | None = "balanced") -> Pipeline:
    return Pipeline(
        [
            ("features", make_domain_robust_feature_union()),
            ("clf", LinearSVC(class_weight=class_weight)),
        ]
    )


def make_logreg_pipeline(class_weight: str | None = "balanced") -> Pipeline:
    return Pipeline(
        [
            ("features", make_feature_union()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    class_weight=class_weight,
                ),
            ),
        ]
    )


def make_domain_robust_logreg_pipeline(class_weight: str | None = "balanced") -> Pipeline:
    return Pipeline(
        [
            ("features", make_domain_robust_feature_union()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    class_weight=class_weight,
                ),
            ),
        ]
    )
def make_minilm_logreg_pipeline(
    encoder_name: str = "all-MiniLM-L6-v2",
    class_weight: str | None = "balanced",
) -> MiniLMLogisticClassifier:
    return MiniLMLogisticClassifier(
        encoder_name=encoder_name,
        class_weight=class_weight,
    )

class EmbeddingLogisticClassifier(BaseEstimator, ClassifierMixin):
    """Logistic head over compact transformer embeddings.

    Set ``encoder_name`` to a local MiniLM/SmolLM-style encoder snapshot through
    the constructor or ``BLOOM_ENCODER_NAME`` in the evaluator. The encoder is
    offline-first and falls back to hashing if a transformer checkpoint is not
    available, preserving reproducible CPU-only execution.
    """

    def __init__(
        self,
        encoder_name: str = "all-MiniLM-L6-v2",
        class_weight: str | None = "balanced",
        max_iter: int = 2000,
        batch_size: int = 64,
        n_features: int = 384,
        local_files_only: bool = True,
    ) -> None:
        self.encoder_name = encoder_name
        self.class_weight = class_weight
        self.max_iter = int(max_iter)
        self.batch_size = int(batch_size)
        self.n_features = int(n_features)
        self.local_files_only = bool(local_files_only)

    def fit(self, X: Sequence[str], y: Sequence[str]):
        self.encoder_ = StableTextEncoder(
            self.encoder_name,
            device="cpu",
            local_files_only=self.local_files_only,
            n_features=self.n_features,
        )
        X_emb = self.encoder_.encode(
            list(X),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        self.clf_ = LogisticRegression(
            max_iter=self.max_iter,
            solver="lbfgs",
            class_weight=self.class_weight,
        )
        self.clf_.fit(X_emb, list(y))
        self.classes_ = self.clf_.classes_
        return self

    def _encode(self, X: Sequence[str]) -> np.ndarray:
        if not hasattr(self, "encoder_"):
            raise RuntimeError("classifier is not fitted")
        return self.encoder_.encode(
            list(X),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def predict(self, X: Sequence[str]) -> np.ndarray:
        return self.clf_.predict(self._encode(X))

    def predict_proba(self, X: Sequence[str]) -> np.ndarray:
        return self.clf_.predict_proba(self._encode(X))


def make_embedding_logreg_classifier(
    encoder_name: str = "all-MiniLM-L6-v2",
    class_weight: str | None = "balanced",
) -> EmbeddingLogisticClassifier:
    return EmbeddingLogisticClassifier(
        encoder_name=encoder_name,
        class_weight=class_weight,
    )


class HierarchicalBloomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, class_weight: str | None = "balanced") -> None:
        self.class_weight = class_weight
        self.root_ = make_linear_svm_pipeline(class_weight=class_weight)
        self.lower_ = make_linear_svm_pipeline(class_weight=class_weight)
        self.higher_ = make_linear_svm_pipeline(class_weight=class_weight)
        self.classes_ = np.array(BLOOM_LEVELS)

    def fit(self, X: List[str], y: List[str]):
        X_list = list(X)
        y_list = list(y)
        root_y = ["lower" if label in LOWER_ORDER else "higher" for label in y_list]
        self.root_ = make_linear_svm_pipeline(class_weight=self.class_weight)
        self.root_.fit(X_list, root_y)

        lower_X = [x for x, label in zip(X_list, y_list) if label in LOWER_ORDER]
        lower_y = [label for label in y_list if label in LOWER_ORDER]
        higher_X = [x for x, label in zip(X_list, y_list) if label in HIGHER_ORDER]
        higher_y = [label for label in y_list if label in HIGHER_ORDER]

        self.lower_ = make_linear_svm_pipeline(class_weight=self.class_weight)
        self.higher_ = make_linear_svm_pipeline(class_weight=self.class_weight)
        self.lower_.fit(lower_X, lower_y)
        self.higher_.fit(higher_X, higher_y)
        return self

    def predict(self, X: List[str]) -> np.ndarray:
        root_scores = self.root_.decision_function(X)
        p_lower = 1 / (1 + np.exp(-root_scores))  # sigmoid
        p_higher = 1 - p_lower

        out = []

        lower_preds = self.lower_.predict(X)
        higher_preds = self.higher_.predict(X)

        for i in range(len(X)):
            if p_lower[i] >= 0.5:
                out.append(str(lower_preds[i]))
            else:
                out.append(str(higher_preds[i]))

        return np.array(out)

    def predict_proba(self, X: List[str]) -> np.ndarray:
        # -------- root routing (lower vs higher Bloom) --------
        root_scores = self.root_.decision_function(X)
        root_scores = np.asarray(root_scores, dtype=np.float64).reshape(-1)

        # numerically stable sigmoid
        p_higher = 1.0 / (1.0 + np.exp(-np.clip(root_scores, -20, 20)))
        p_lower = 1.0 - p_higher

        # -------- branch probabilities --------
        lower_scores = np.asarray(self.lower_.decision_function(X), dtype=np.float64)
        higher_scores = np.asarray(self.higher_.decision_function(X), dtype=np.float64)

        if lower_scores.ndim == 1:
            lower_scores = lower_scores[:, None]
        if higher_scores.ndim == 1:
            higher_scores = higher_scores[:, None]

        lower_probs = _softmax_rows(lower_scores)
        higher_probs = _softmax_rows(higher_scores)

        # -------- combine into full distribution --------
        out = np.zeros((len(X), len(BLOOM_LEVELS)), dtype=np.float64)

        # lower-order (Remember, Understand, Apply)
        out[:, 0:3] = p_lower[:, None] * lower_probs

        # higher-order (Analyze, Evaluate, Create)
        out[:, 3:6] = p_higher[:, None] * higher_probs

        # -------- safety normalization (important for papers) --------
        row_sum = out.sum(axis=1, keepdims=True)
        out = out / np.clip(row_sum, 1e-12, None)

        return out


class OrdinalThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        class_weight: str | None = "balanced",
        levels: Sequence[str] | None = None,
    ) -> None:
        self.class_weight = class_weight
        self.levels = levels
        self.vectorizer_ = make_feature_union()
        self.threshold_models_: List[LogisticRegression] = []
        self.classes_ = np.array(list(self.levels) if self.levels is not None else BLOOM_LEVELS)

    def fit(self, X: List[str], y: List[str]):
        X_list = list(X)
        levels = list(self.levels) if self.levels is not None else BLOOM_LEVELS
        y_idx = np.array([levels.index(label) for label in y], dtype=np.int32)
        self.vectorizer_ = make_feature_union()
        X_vec = self.vectorizer_.fit_transform(X_list)
        self.threshold_models_ = []
        for threshold in range(len(levels) - 1):
            binary = (y_idx > threshold).astype(np.int32)
            clf = LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                class_weight=self.class_weight,
            )
            clf.fit(X_vec, binary)
            self.threshold_models_.append(clf)
        self.classes_ = np.array(levels)
        return self

    def predict_proba(self, X: List[str]) -> np.ndarray:
        X_vec = self.vectorizer_.transform(list(X))
        threshold_probs = []
        for clf in self.threshold_models_:
            p = clf.predict_proba(X_vec)[:, 1]
            threshold_probs.append(p)
        cum = np.vstack(threshold_probs).T
        # Enforce monotonic decreasing P(y > k)
        cum = np.minimum.accumulate(cum, axis=1)

        levels = list(self.classes_)
        out = np.zeros((len(X), len(levels)), dtype=np.float64)
        out[:, 0] = 1.0 - cum[:, 0]
        for idx in range(1, len(levels) - 1):
            out[:, idx] = np.clip(cum[:, idx - 1] - cum[:, idx], 0.0, 1.0)
        out[:, -1] = np.clip(cum[:, -1], 0.0, 1.0)
        out = out / np.clip(out.sum(axis=1, keepdims=True), 1e-9, None)
        return out

    def predict(self, X: List[str]) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        levels = list(self.classes_)
        return np.array([levels[i] for i in idx])

class MiniLMLogisticClassifier(BaseEstimator, ClassifierMixin):
    """
    CPU-safe transformer baseline:
    Frozen MiniLM embeddings + Logistic Regression head.
    Used as a neural upper-bound comparator for TF-IDF models.
    """

    def __init__(
        self,
        encoder_name: str = "all-MiniLM-L6-v2",
        class_weight: str | None = "balanced",
        max_iter: int = 2000,
        batch_size: int = 64,
        local_files_only: bool = True,
    ) -> None:
        self.encoder_name = encoder_name
        self.class_weight = class_weight
        self.max_iter = int(max_iter)
        self.batch_size = int(batch_size)
        self.local_files_only = bool(local_files_only)

    def fit(self, X: Sequence[str], y: Sequence[str]):
        self.encoder_ = StableTextEncoder(
            self.encoder_name,
            device="cpu",
            local_files_only=self.local_files_only,
        )

        X_emb = self.encoder_.encode(
            list(X),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        self.clf_ = LogisticRegression(
            max_iter=self.max_iter,
            solver="lbfgs",
            class_weight=self.class_weight,
        )

        self.clf_.fit(X_emb, list(y))
        self.classes_ = self.clf_.classes_
        return self

    def _encode(self, X: Sequence[str]) -> np.ndarray:
        return self.encoder_.encode(
            list(X),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def predict(self, X: Sequence[str]) -> np.ndarray:
        return self.clf_.predict(self._encode(X))

    def predict_proba(self, X: Sequence[str]) -> np.ndarray:
        return self.clf_.predict_proba(self._encode(X))
def _softmax_rows(scores: np.ndarray) -> np.ndarray:
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / np.clip(exp_scores.sum(axis=1, keepdims=True), 1e-9, None)
