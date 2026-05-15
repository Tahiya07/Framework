from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_MODEL_PATH = Path("models/federated_privacy_guard.json")
DEFAULT_SEED = 42


@dataclass
class PrivacyTrainingExample:
    text: str
    label: int
    client_id: str
    role: str = "teacher"
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class FederatedPrivacyConfig:
    rounds: int = 25
    local_epochs: int = 3
    learning_rate: float = 0.18
    l2: float = 1e-4
    n_features: int = 2048
    client_fraction: float = 1.0
    min_clients: int = 2
    clip_norm: float = 2.5
    dp_noise: float = 0.2
    dp_delta: float = 1e-5
    threshold: float = 0.55
    seed: int = DEFAULT_SEED


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _ngrams(tokens: Sequence[str], n: int) -> Iterable[str]:
    if len(tokens) < n:
        return []
    return (" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _stable_index(token: str, n_features: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % n_features


class PrivacyFeatureHasher:
    """Shared fixed featurizer for federated privacy-risk training.

    It exports no vocabulary learned from teacher data. Every client hashes text
    locally into the same feature space, so the server receives only numeric
    model updates.
    """

    def __init__(self, n_features: int = 2048) -> None:
        self.n_features = int(n_features)

    def transform_one(self, text: str) -> Dict[int, float]:
        tokens = _tokens(text)
        feats: Dict[int, float] = {}
        for tok in tokens:
            idx = _stable_index(f"tok:{tok}", self.n_features)
            feats[idx] = feats.get(idx, 0.0) + 1.0
        for n in (2, 3):
            for gram in _ngrams(tokens, n):
                idx = _stable_index(f"ng{n}:{gram}", self.n_features)
                feats[idx] = feats.get(idx, 0.0) + 0.75

        # Small auditable intent indicators. These are not private vocabulary.
        lower = (text or "").lower()
        indicators = {
            "has_exam": bool(re.search(r"\b(exam|question paper|paper)\b", lower)),
            "has_protected": bool(re.search(r"\b(protected|uploaded|hidden|previously uploaded)\b", lower)),
            "has_reconstruct": bool(re.search(r"\b(reconstruct|recover|reveal|show|quote|verbatim|exact|copy)\b", lower)),
            "has_paraphrase": bool(re.search(r"\b(paraphrase|rewrite|in your own words|same concepts|clone)\b", lower)),
            "has_teacher_moderate": bool(re.search(r"\b(moderate|classify|bloom|revise|alignment)\b", lower)),
        }
        for name, active in indicators.items():
            if active:
                idx = _stable_index(f"indicator:{name}", self.n_features)
                feats[idx] = feats.get(idx, 0.0) + 2.0

        norm = math.sqrt(sum(v * v for v in feats.values()))
        if norm > 0.0:
            feats = {k: v / norm for k, v in feats.items()}
        return feats


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-min(50.0, x))
        return 1.0 / (1.0 + z)
    z = math.exp(max(-50.0, x))
    return z / (1.0 + z)


def _predict_raw(weights: Sequence[float], bias: float, feats: Dict[int, float]) -> float:
    return bias + sum(weights[idx] * value for idx, value in feats.items())


def _clip_vector(delta_w: List[float], delta_b: float, clip_norm: float) -> Tuple[List[float], float]:
    norm = math.sqrt(sum(v * v for v in delta_w) + delta_b * delta_b)
    if norm <= clip_norm or norm == 0.0:
        return delta_w, delta_b
    scale = clip_norm / norm
    return [v * scale for v in delta_w], delta_b * scale


def estimate_dp_epsilon(
    *,
    rounds: int,
    sampling_rate: float,
    noise_multiplier: float,
    delta: float,
) -> float | None:
    """Conservative closed-form DP-SGD epsilon estimate.

    This is an auditable deployment estimate, not a replacement for a full RDP
    accountant. It is intentionally conservative for reporting and gating:

        epsilon ~= q * sqrt(2 * T * log(1/delta)) / sigma

    where q is client sampling rate, T is communication rounds, and sigma is
    the Gaussian noise multiplier applied after clipping.
    """
    if noise_multiplier <= 0.0 or delta <= 0.0 or delta >= 1.0:
        return None
    return float(
        sampling_rate
        * math.sqrt(2.0 * max(1, rounds) * math.log(1.0 / delta))
        / noise_multiplier
    )


def estimate_rdp_epsilon(
    *,
    rounds: int,
    noise_multiplier: float,
    delta: float,
    orders: Sequence[int] | None = None,
) -> Dict[str, float | int | None]:
    """Conservative RDP accountant for full-participation Gaussian updates.

    For each Renyi order alpha, Gaussian mechanism RDP is alpha / (2 sigma^2).
    We compose linearly over rounds and convert to (epsilon, delta)-DP via:

        epsilon = rdp + log(1/delta) / (alpha - 1)

    This ignores client subsampling amplification, so it is conservative for
    sampled settings. It is suitable for transparent prototype reporting.
    """
    if noise_multiplier <= 0.0 or delta <= 0.0 or delta >= 1.0:
        return {"epsilon": None, "best_order": None}
    candidates = list(orders or range(2, 65))
    best_eps = None
    best_order = None
    for alpha in candidates:
        rdp = max(1, rounds) * (float(alpha) / (2.0 * noise_multiplier * noise_multiplier))
        eps = rdp + math.log(1.0 / delta) / float(alpha - 1)
        if best_eps is None or eps < best_eps:
            best_eps = eps
            best_order = alpha
    return {"epsilon": float(best_eps), "best_order": int(best_order) if best_order else None}


class FederatedPrivacyGuardModel:
    def __init__(
        self,
        weights: Sequence[float],
        bias: float,
        *,
        threshold: float,
        n_features: int,
        metadata: Dict[str, object] | None = None,
    ) -> None:
        self.weights = [float(v) for v in weights]
        self.bias = float(bias)
        self.threshold = float(threshold)
        self.n_features = int(n_features)
        self.metadata = dict(metadata or {})
        self.hasher = PrivacyFeatureHasher(self.n_features)

    def risk_score(self, text: str) -> float:
        feats = self.hasher.transform_one(text)
        return float(_sigmoid(_predict_raw(self.weights, self.bias, feats)))

    def blocks(self, text: str) -> bool:
        return self.risk_score(text) >= self.threshold

    def to_dict(self) -> Dict[str, object]:
        return {
            "format": "federated_privacy_guard_v1",
            "weights": self.weights,
            "bias": self.bias,
            "threshold": self.threshold,
            "n_features": self.n_features,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "FederatedPrivacyGuardModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "federated_privacy_guard_v1":
            raise ValueError(f"Unsupported federated privacy model format: {payload.get('format')!r}")
        return cls(
            payload["weights"],
            float(payload["bias"]),
            threshold=float(payload["threshold"]),
            n_features=int(payload["n_features"]),
            metadata=dict(payload.get("metadata") or {}),
        )


def _local_train_update(
    base_weights: Sequence[float],
    base_bias: float,
    examples: Sequence[PrivacyTrainingExample],
    hasher: PrivacyFeatureHasher,
    config: FederatedPrivacyConfig,
    rng: random.Random,
) -> Tuple[List[float], float]:
    weights = [float(v) for v in base_weights]
    bias = float(base_bias)
    rows = list(examples)

    for _ in range(config.local_epochs):
        rng.shuffle(rows)
        for row in rows:
            feats = hasher.transform_one(row.text)
            pred = _sigmoid(_predict_raw(weights, bias, feats))
            error = pred - float(row.label)
            for idx, value in feats.items():
                grad = error * value + config.l2 * weights[idx]
                weights[idx] -= config.learning_rate * grad
            bias -= config.learning_rate * error

    delta_w = [w - bw for w, bw in zip(weights, base_weights)]
    delta_b = bias - base_bias
    return _clip_vector(delta_w, delta_b, config.clip_norm)


def train_federated_privacy_guard(
    examples: Sequence[PrivacyTrainingExample],
    config: FederatedPrivacyConfig | None = None,
) -> Tuple[FederatedPrivacyGuardModel, Dict[str, object]]:
    config = config or FederatedPrivacyConfig()
    rng = random.Random(config.seed)
    by_client: Dict[str, List[PrivacyTrainingExample]] = {}
    for row in examples:
        by_client.setdefault(str(row.client_id), []).append(row)
    if len(by_client) < config.min_clients:
        raise ValueError(f"Need at least {config.min_clients} clients for federated privacy training")

    hasher = PrivacyFeatureHasher(config.n_features)
    weights = [0.0] * config.n_features
    bias = 0.0
    clients = sorted(by_client)
    history: List[Dict[str, float]] = []

    for round_idx in range(config.rounds):
        shuffled = clients[:]
        rng.shuffle(shuffled)
        take = max(config.min_clients, int(math.ceil(len(clients) * config.client_fraction)))
        selected = shuffled[: min(len(shuffled), take)]

        weighted_updates: List[Tuple[float, List[float], float]] = []
        total_rows = 0
        for client_id in selected:
            rows = by_client[client_id]
            delta_w, delta_b = _local_train_update(weights, bias, rows, hasher, config, rng)
            if config.dp_noise > 0.0:
                sigma = config.dp_noise * config.clip_norm
                delta_w = [v + rng.gauss(0.0, sigma) for v in delta_w]
                delta_b = delta_b + rng.gauss(0.0, sigma)
            weighted_updates.append((float(len(rows)), delta_w, delta_b))
            total_rows += len(rows)

        if total_rows == 0:
            continue
        agg_w = [0.0] * config.n_features
        agg_b = 0.0
        for count, delta_w, delta_b in weighted_updates:
            alpha = count / float(total_rows)
            for i, value in enumerate(delta_w):
                agg_w[i] += alpha * value
            agg_b += alpha * delta_b
        weights = [w + dw for w, dw in zip(weights, agg_w)]
        bias += agg_b

        if round_idx == 0 or (round_idx + 1) % 5 == 0 or round_idx + 1 == config.rounds:
            model = FederatedPrivacyGuardModel(
                weights,
                bias,
                threshold=config.threshold,
                n_features=config.n_features,
            )
            metrics = evaluate_privacy_model(model, examples)
            history.append(
                {
                    "round": float(round_idx + 1),
                    "accuracy": metrics["accuracy"],
                    "attack_block_rate": metrics["attack_block_rate"],
                    "benign_allow_rate": metrics["benign_allow_rate"],
                }
            )

    metadata = {
        "architecture": "federated_privacy_guard",
        "num_clients": len(clients),
        "rounds": config.rounds,
        "local_epochs": config.local_epochs,
        "client_fraction": config.client_fraction,
        "clip_norm": config.clip_norm,
        "dp_noise": config.dp_noise,
        "dp_delta": config.dp_delta,
        "dp_epsilon_estimate": estimate_dp_epsilon(
            rounds=config.rounds,
            sampling_rate=min(1.0, max(config.min_clients / max(1, len(clients)), config.client_fraction)),
            noise_multiplier=config.dp_noise,
            delta=config.dp_delta,
        ),
        "rdp_accountant": estimate_rdp_epsilon(
            rounds=config.rounds,
            noise_multiplier=config.dp_noise,
            delta=config.dp_delta,
        ),
        "dp_accounting_note": (
            "RDP accountant is conservative for full-participation Gaussian clipped updates "
            "and ignores any privacy amplification from client subsampling."
        ),
        "threshold": config.threshold,
        "n_features": config.n_features,
        "server_receives": "secure-aggregation-compatible clipped/noised aggregate model updates only",
        "server_does_not_store": [
            "raw teacher questions",
            "raw protected exam text",
            "per-teacher examples",
            "per-teacher metrics",
            "client vocabularies",
        ],
        "history": history,
    }
    model = FederatedPrivacyGuardModel(
        weights,
        bias,
        threshold=config.threshold,
        n_features=config.n_features,
        metadata=metadata,
    )
    return model, metadata


def evaluate_privacy_model(
    model: FederatedPrivacyGuardModel,
    examples: Sequence[PrivacyTrainingExample],
) -> Dict[str, float]:
    rows = list(examples)
    if not rows:
        return {"accuracy": 0.0, "attack_block_rate": 0.0, "benign_allow_rate": 0.0}
    correct = 0
    attacks = []
    benign = []
    for row in rows:
        pred = 1 if model.blocks(row.text) else 0
        correct += int(pred == int(row.label))
        if int(row.label) == 1:
            attacks.append(pred)
        else:
            benign.append(pred)
    return {
        "accuracy": correct / len(rows),
        "attack_block_rate": sum(attacks) / max(1, len(attacks)),
        "benign_allow_rate": sum(1 for pred in benign if pred == 0) / max(1, len(benign)),
    }


def load_default_model(path: str | Path = DEFAULT_MODEL_PATH) -> FederatedPrivacyGuardModel | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return FederatedPrivacyGuardModel.load(path)
    except Exception:
        return None
