from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from bloom.bloom_models import BloomCueTransformer, make_domain_robust_feature_union
from classifier import BLOOM_LEVELS, _normalise_bloom


SEED = 42
DEFAULT_DATASET = Path("data/figshare_combined_dataset.csv")
DEFAULT_OUTPUT_DIR = Path("results/federated_bloom")
DEFAULT_MODEL_DIR = Path("models")


def _clean_text(value: object) -> str:
    return " ".join(str(value).strip().split())


def _find_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    lowered = {str(c).strip().lower(): str(c) for c in columns}
    for candidate in candidates:
        found = lowered.get(candidate.lower())
        if found is not None:
            return found
    return None


def load_bloom_dataset(
    path: Path,
    *,
    teacher_col: str | None = None,
    role_col: str | None = None,
    teacher_roles: Sequence[str] = ("teacher", "instructor", "faculty", "moderator"),
) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    question_col = _find_column(raw.columns, ["question", "questions", "text", "QUESTION"])
    label_col = _find_column(raw.columns, ["bloom_level", "bloom", "label", "BT LEVEL", "level"])

    if question_col is None or label_col is None:
        raise ValueError(
            f"Could not detect question/label columns in {path}. "
            f"Columns: {list(raw.columns)}"
        )

    df = pd.DataFrame(
        {
            "question": raw[question_col].map(_clean_text),
            "bloom_level": raw[label_col].map(_normalise_bloom),
        }
    )

    if teacher_col and teacher_col in raw.columns:
        df["teacher_id"] = raw[teacher_col].astype(str).map(_clean_text)
    else:
        df["teacher_id"] = ""

    if role_col and role_col in raw.columns:
        roles = raw[role_col].astype(str).str.strip().str.lower()
        allowed = {str(r).strip().lower() for r in teacher_roles}
        df = df[roles.isin(allowed)].copy()

    df = df.dropna(subset=["question", "bloom_level"])
    df = df[df["question"].str.len() > 0]
    df = df[df["bloom_level"].isin(BLOOM_LEVELS)].copy()

    df = df.drop_duplicates(subset=["question", "bloom_level"])
    conflicts = df.groupby("question")["bloom_level"].nunique()
    conflicted_questions = set(conflicts[conflicts > 1].index)
    df = df[~df["question"].isin(conflicted_questions)]
    df = df.drop_duplicates(subset=["question"]).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(f"No usable Bloom rows found in {path}")

    return df


def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, temp = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["bloom_level"],
    )
    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=SEED,
        stratify=temp["bloom_level"],
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def ordinal_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, float]:
    label_to_idx = {label: i for i, label in enumerate(BLOOM_LEVELS)}
    true_idx = np.array([label_to_idx[y] for y in y_true], dtype=np.int32)
    pred_idx = np.array([label_to_idx[y] for y in y_pred], dtype=np.int32)
    distance = np.abs(true_idx - pred_idx)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "mean_ordinal_error": float(distance.mean()),
        "within_one_level_accuracy": float((distance <= 1).mean()),
        "severe_error_rate": float((distance >= 2).mean()),
    }


def evaluate_model(model: object, X: Sequence[str], y: Sequence[str]) -> Dict[str, object]:
    pred = list(model.predict(list(X)))
    metrics = ordinal_metrics(y, pred)
    metrics["classification_report"] = classification_report(
        y,
        pred,
        labels=BLOOM_LEVELS,
        output_dict=True,
        zero_division=0,
    )
    metrics["confusion_matrix"] = confusion_matrix(y, pred, labels=BLOOM_LEVELS).tolist()
    return metrics


def build_centralized_candidates() -> Dict[str, object]:
    feature_union = make_domain_robust_feature_union()
    return {
        "domain_robust_logreg": Pipeline(
            [
                ("features", feature_union),
                (
                    "clf",
                    LogisticRegression(
                        C=2.0,
                        max_iter=3000,
                        solver="lbfgs",
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "domain_robust_svm": Pipeline(
            [
                ("features", make_domain_robust_feature_union()),
                ("clf", LinearSVC(C=1.5, class_weight="balanced")),
            ]
        ),
        "wide_tfidf_logreg": Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            (
                                "word",
                                TfidfVectorizer(
                                    lowercase=True,
                                    strip_accents="unicode",
                                    ngram_range=(1, 3),
                                    min_df=1,
                                    max_features=90000,
                                    sublinear_tf=True,
                                ),
                            ),
                            (
                                "char",
                                TfidfVectorizer(
                                    analyzer="char_wb",
                                    lowercase=True,
                                    ngram_range=(3, 6),
                                    min_df=1,
                                    max_features=90000,
                                    sublinear_tf=True,
                                ),
                            ),
                            ("bloom_cues", BloomCueTransformer(normalize=True)),
                        ]
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=3.0,
                        max_iter=3000,
                        solver="lbfgs",
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
    }


def cross_validate_candidates(
    models: Dict[str, object],
    X: Sequence[str],
    y: Sequence[str],
    *,
    n_splits: int = 5,
) -> Dict[str, Dict[str, float]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    X_arr = np.asarray(list(X), dtype=object)
    y_arr = np.asarray(list(y), dtype=object)
    results: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        fold_f1: List[float] = []
        fold_acc: List[float] = []
        for train_idx, val_idx in skf.split(X_arr, y_arr):
            candidate = clone(model)
            candidate.fit(X_arr[train_idx].tolist(), y_arr[train_idx].tolist())
            pred = candidate.predict(X_arr[val_idx].tolist())
            fold_f1.append(float(f1_score(y_arr[val_idx], pred, average="macro")))
            fold_acc.append(float(accuracy_score(y_arr[val_idx], pred)))
        results[name] = {
            "macro_f1_mean": float(np.mean(fold_f1)),
            "macro_f1_std": float(np.std(fold_f1)),
            "accuracy_mean": float(np.mean(fold_acc)),
            "accuracy_std": float(np.std(fold_acc)),
        }
    return results


class PrivacyPreservingBloomFeaturizer:
    """Stateless shared featurizer for federated Bloom moderation.

    HashingVectorizer avoids exporting a vocabulary learned from any teacher's
    private question bank. BloomCueTransformer contributes small, domain-light
    cognitive-demand features that do not reveal teacher-authored text.
    """

    def __init__(self, n_features: int = 2**17) -> None:
        self.word = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 3),
            norm="l2",
        )
        self.char = HashingVectorizer(
            analyzer="char_wb",
            n_features=n_features // 2,
            alternate_sign=False,
            lowercase=True,
            ngram_range=(3, 6),
            norm="l2",
        )
        self.cues = BloomCueTransformer(normalize=True).fit([])

    def transform(self, texts: Sequence[str]):
        texts = list(texts)
        return sparse.hstack(
            [
                self.word.transform(texts),
                self.char.transform(texts),
                self.cues.transform(texts),
            ],
            format="csr",
        )


@dataclass
class FederatedConfig:
    rounds: int = 30
    local_epochs: int = 2
    client_fraction: float = 1.0
    min_clients: int = 4
    pseudo_clients: int = 8
    alpha: float = 1e-5
    differential_privacy_noise: float = 0.0
    clip_norm: float = 8.0


def assign_teacher_clients(train: pd.DataFrame, *, pseudo_clients: int) -> pd.DataFrame:
    out = train.copy()
    if out["teacher_id"].str.len().sum() > 0 and out["teacher_id"].nunique() >= 2:
        out["client_id"] = out["teacher_id"]
        return out

    # Public Figshare files do not expose teacher identities. For architecture
    # testing, create deterministic proxy clients that preserve label balance
    # without writing per-client rows or metrics to disk.
    client_ids: List[str] = []
    counters = {label: 0 for label in BLOOM_LEVELS}
    for label in out["bloom_level"].tolist():
        idx = counters[label] % pseudo_clients
        counters[label] += 1
        client_ids.append(f"teacher_client_{idx:02d}")
    out["client_id"] = client_ids
    return out


def _balanced_sample_weights(y: Sequence[str]) -> np.ndarray:
    labels = list(y)
    counts = {label: max(1, labels.count(label)) for label in BLOOM_LEVELS}
    total = float(len(labels))
    k = float(len(BLOOM_LEVELS))
    return np.array([total / (k * counts[label]) for label in labels], dtype=np.float64)


def _init_classifier(alpha: float, X_seed, y_seed: Sequence[str]) -> SGDClassifier:
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        fit_intercept=True,
        random_state=SEED,
        learning_rate="optimal",
        average=False,
    )
    clf.partial_fit(X_seed, list(y_seed), classes=np.array(BLOOM_LEVELS, dtype=object))
    return clf


def _copy_classifier(src: SGDClassifier, alpha: float) -> SGDClassifier:
    dst = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        fit_intercept=True,
        random_state=SEED,
        learning_rate="optimal",
        average=False,
    )
    dst.classes_ = src.classes_.copy()
    dst.coef_ = src.coef_.copy()
    dst.intercept_ = src.intercept_.copy()
    if hasattr(src, "t_"):
        dst.t_ = src.t_
    if hasattr(src, "n_iter_"):
        dst.n_iter_ = src.n_iter_
    return dst


def _clip_update(update: np.ndarray, clip_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(update))
    if norm <= clip_norm or norm == 0.0:
        return update
    return update * (clip_norm / norm)


class FederatedBloomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, featurizer: PrivacyPreservingBloomFeaturizer, clf: SGDClassifier) -> None:
        self.featurizer = featurizer
        self.clf = clf
        self.classes_ = np.array(BLOOM_LEVELS, dtype=object)

    def predict(self, X: Sequence[str]) -> np.ndarray:
        return self.clf.predict(self.featurizer.transform(list(X)))

    def predict_proba(self, X: Sequence[str]) -> np.ndarray:
        return self.clf.predict_proba(self.featurizer.transform(list(X)))


def train_federated_bloom(
    train: pd.DataFrame,
    val: pd.DataFrame,
    config: FederatedConfig,
) -> Tuple[FederatedBloomClassifier, Dict[str, object]]:
    rng = np.random.default_rng(SEED)
    train = assign_teacher_clients(train, pseudo_clients=config.pseudo_clients)
    clients = sorted(train["client_id"].unique().tolist())
    if len(clients) < config.min_clients:
        train = assign_teacher_clients(train.assign(teacher_id=""), pseudo_clients=config.min_clients)
        clients = sorted(train["client_id"].unique().tolist())

    featurizer = PrivacyPreservingBloomFeaturizer()
    X_seed = featurizer.transform(train["question"].head(len(BLOOM_LEVELS)).tolist())
    y_seed = train["bloom_level"].head(len(BLOOM_LEVELS)).tolist()
    if len(set(y_seed)) < len(BLOOM_LEVELS):
        seed_rows = train.groupby("bloom_level", group_keys=False).head(1)
        X_seed = featurizer.transform(seed_rows["question"].tolist())
        y_seed = seed_rows["bloom_level"].tolist()

    global_clf = _init_classifier(config.alpha, X_seed, y_seed)
    val_X = val["question"].tolist()
    val_y = val["bloom_level"].tolist()
    round_history: List[Dict[str, float]] = []

    for round_idx in range(config.rounds):
        shuffled = clients.copy()
        rng.shuffle(shuffled)
        take = max(config.min_clients, int(math.ceil(len(clients) * config.client_fraction)))
        selected = shuffled[: min(len(shuffled), take)]

        coef_base = global_clf.coef_.copy()
        intercept_base = global_clf.intercept_.copy()
        coef_updates = []
        intercept_updates = []
        weights = []

        for client_id in selected:
            part = train[train["client_id"] == client_id]
            X_client = featurizer.transform(part["question"].tolist())
            y_client = part["bloom_level"].tolist()
            sample_weight = _balanced_sample_weights(y_client)

            local = _copy_classifier(global_clf, config.alpha)
            for _ in range(config.local_epochs):
                order = rng.permutation(len(y_client))
                local.partial_fit(
                    X_client[order],
                    np.asarray(y_client, dtype=object)[order],
                    classes=np.array(BLOOM_LEVELS, dtype=object),
                    sample_weight=sample_weight[order],
                )

            coef_delta = _clip_update(local.coef_ - coef_base, config.clip_norm)
            intercept_delta = _clip_update(local.intercept_ - intercept_base, config.clip_norm)

            if config.differential_privacy_noise > 0:
                coef_delta = coef_delta + rng.normal(
                    0.0,
                    config.differential_privacy_noise * config.clip_norm,
                    size=coef_delta.shape,
                )
                intercept_delta = intercept_delta + rng.normal(
                    0.0,
                    config.differential_privacy_noise * config.clip_norm,
                    size=intercept_delta.shape,
                )

            coef_updates.append(coef_delta)
            intercept_updates.append(intercept_delta)
            weights.append(float(len(part)))

        weight_arr = np.asarray(weights, dtype=np.float64)
        weight_arr = weight_arr / np.clip(weight_arr.sum(), 1e-12, None)

        # Secure-aggregation simulation: only weighted update sums are applied;
        # no per-teacher coefficients, examples, or metrics are persisted.
        global_clf.coef_ = coef_base + np.tensordot(weight_arr, np.stack(coef_updates), axes=(0, 0))
        global_clf.intercept_ = intercept_base + np.tensordot(
            weight_arr,
            np.stack(intercept_updates),
            axes=(0, 0),
        )

        if (round_idx + 1) == 1 or (round_idx + 1) % 5 == 0 or (round_idx + 1) == config.rounds:
            model = FederatedBloomClassifier(featurizer, global_clf)
            metrics = ordinal_metrics(val_y, model.predict(val_X))
            round_history.append(
                {
                    "round": float(round_idx + 1),
                    "val_accuracy": metrics["accuracy"],
                    "val_macro_f1": metrics["macro_f1"],
                    "val_severe_error_rate": metrics["severe_error_rate"],
                }
            )

    metadata = {
        "num_clients": len(clients),
        "rounds": config.rounds,
        "local_epochs": config.local_epochs,
        "client_fraction": config.client_fraction,
        "differential_privacy_noise": config.differential_privacy_noise,
        "clip_norm": config.clip_norm,
        "privacy_notes": [
            "Raw teacher questions stay inside client partitions.",
            "The server receives only clipped aggregate weight updates.",
            "HashingVectorizer avoids exporting a teacher-specific vocabulary.",
            "Per-teacher metrics and examples are intentionally not written.",
        ],
        "round_history": round_history,
    }
    return FederatedBloomClassifier(featurizer, global_clf), metadata


def read_prior_prompt_metrics() -> Dict[str, object]:
    candidates = [
        Path("evaluation_outputs/results_table.csv"),
        Path("results/unified_results_table.csv"),
    ]
    out: Dict[str, object] = {}
    for path in candidates:
        if not path.is_file():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "Model" in df.columns:
            row = df[df["Model"].astype(str).str.upper() == "QWEN"]
            if not row.empty:
                out["prompt_qwen"] = {
                    "source": str(path),
                    "accuracy": float(row.iloc[0]["Accuracy"]),
                    "macro_f1": float(row.iloc[0]["Macro-F1"]),
                    "severe_error_rate": float(row.iloc[0]["Severe Error"]),
                }
    return out


def train(args: argparse.Namespace) -> Dict[str, object]:
    random.seed(SEED)
    np.random.seed(SEED)

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_bloom_dataset(
        dataset_path,
        teacher_col=args.teacher_col,
        role_col=args.role_col,
        teacher_roles=tuple(args.teacher_roles),
    )
    train_df, val_df, test_df = split_dataset(df)

    X_train = train_df["question"].tolist()
    y_train = train_df["bloom_level"].tolist()
    X_val = val_df["question"].tolist()
    y_val = val_df["bloom_level"].tolist()
    X_test = test_df["question"].tolist()
    y_test = test_df["bloom_level"].tolist()

    centralized = build_centralized_candidates()
    cv_results = cross_validate_candidates(centralized, X_train, y_train, n_splits=args.cv_folds)

    validation_results: Dict[str, Dict[str, object]] = {}
    fitted_models: Dict[str, object] = {}
    for name, model in centralized.items():
        fitted = clone(model)
        fitted.fit(X_train, y_train)
        validation_results[name] = evaluate_model(fitted, X_val, y_val)
        fitted_models[name] = fitted

    fed_config = FederatedConfig(
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        client_fraction=args.client_fraction,
        min_clients=args.min_clients,
        pseudo_clients=args.pseudo_clients,
        alpha=args.alpha,
        differential_privacy_noise=args.dp_noise,
        clip_norm=args.clip_norm,
    )
    federated_model, federated_metadata = train_federated_bloom(train_df, val_df, fed_config)
    validation_results["federated_teacher_private"] = evaluate_model(federated_model, X_val, y_val)
    fitted_models["federated_teacher_private"] = federated_model

    selected_name = max(
        validation_results,
        key=lambda name: (
            float(validation_results[name]["macro_f1"]),
            float(validation_results[name]["accuracy"]),
            -float(validation_results[name]["severe_error_rate"]),
        ),
    )

    if selected_name == "federated_teacher_private":
        selected_model = federated_model
    else:
        selected_model = clone(centralized[selected_name])
        X_dev = X_train + X_val
        y_dev = y_train + y_val
        selected_model.fit(X_dev, y_dev)

    t0 = time.time()
    test_metrics = evaluate_model(selected_model, X_test, y_test)
    eval_time = time.time() - t0

    selected_model_path = model_dir / "bloom_teacher_private_best.joblib"
    federated_model_path = model_dir / "bloom_federated_teacher_private.joblib"
    joblib.dump(selected_model, selected_model_path)
    joblib.dump(federated_model, federated_model_path)

    summary = {
        "dataset": str(dataset_path),
        "seed": SEED,
        "label_space": BLOOM_LEVELS,
        "rows": int(len(df)),
        "split": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "label_distribution": df["bloom_level"].value_counts().to_dict(),
        "prior_prompt_metrics": read_prior_prompt_metrics(),
        "centralized_cv_results": cv_results,
        "validation_results": validation_results,
        "federated_architecture": federated_metadata,
        "selected_model": selected_name,
        "selected_model_path": str(selected_model_path),
        "federated_model_path": str(federated_model_path),
        "test_metrics": test_metrics,
        "test_eval_time_sec": eval_time,
        "accuracy_gain_vs_prompt_qwen": None,
    }

    prompt = summary["prior_prompt_metrics"].get("prompt_qwen") if summary["prior_prompt_metrics"] else None
    if prompt:
        summary["accuracy_gain_vs_prompt_qwen"] = float(
            test_metrics["accuracy"] - float(prompt["accuracy"])
        )

    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(
        [
            {
                "model": name,
                "split": "validation",
                **{
                    k: v
                    for k, v in metrics.items()
                    if isinstance(v, (int, float))
                },
            }
            for name, metrics in validation_results.items()
        ]
        + [
            {
                "model": selected_name,
                "split": "test",
                **{
                    k: v
                    for k, v in test_metrics.items()
                    if isinstance(v, (int, float))
                },
            }
        ]
    ).to_csv(output_dir / "metrics.csv", index=False)

    print(json.dumps(summary, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train higher-accuracy Bloom moderation models and a teacher-private "
            "federated architecture for question moderation."
        )
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--teacher-col", default=None, help="Optional column containing teacher/client ids.")
    parser.add_argument("--role-col", default=None, help="Optional role column used to keep teacher-side rows only.")
    parser.add_argument(
        "--teacher-roles",
        nargs="+",
        default=["teacher", "instructor", "faculty", "moderator"],
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--client-fraction", type=float, default=1.0)
    parser.add_argument("--min-clients", type=int, default=4)
    parser.add_argument("--pseudo-clients", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument(
        "--dp-noise",
        type=float,
        default=0.0,
        help="Gaussian noise multiplier for clipped client updates. Increase for stronger DP-style protection.",
    )
    parser.add_argument("--clip-norm", type=float, default=8.0)
    return parser


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
