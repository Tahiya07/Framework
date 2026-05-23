from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from privacy.federated_privacy import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    FederatedPrivacyConfig,
    PrivacyTrainingExample,
    evaluate_privacy_model,
    train_federated_privacy_guard,
)
from privacy.evaluate_privacy_guard import (  # noqa: E402
    PROTECTED_CHUNKS,
    TOPICS,
    _attack_sets,
    _student_benign_sets,
    _teacher_moderation_sets,
)


RESULTS_PATH = ROOT / "results" / "federated_privacy_guard.json"
CSV_PATH = ROOT / "results" / "federated_privacy_guard_rows.csv"


def _client_for_attack(category: str, index: int) -> str:
    # Deterministic proxy clients for local experiments. In deployment, replace
    # these with actual teacher/client IDs and keep rows on each client.
    return f"teacher_client_{index % max(2, len(PROTECTED_CHUNKS)):02d}_{category}"


def build_training_examples() -> List[PrivacyTrainingExample]:
    rows: List[PrivacyTrainingExample] = []
    for category, prompts in _attack_sets().items():
        for idx, item in enumerate(prompts):
            label = 0 if category == "benign_safe_alternative" else 1
            rows.append(
                PrivacyTrainingExample(
                    text=item["query"],
                    label=label,
                    client_id=_client_for_attack(category, idx),
                    role="student",
                    metadata={"category": category, "source": "attack_taxonomy_query"},
                )
            )
            rows.append(
                PrivacyTrainingExample(
                    text=item["answer"],
                    label=label,
                    client_id=_client_for_attack(category, idx),
                    role="student",
                    metadata={"category": category, "source": "attack_taxonomy_answer"},
                )
            )

    for idx, item in enumerate(_student_benign_sets()):
        rows.append(
            PrivacyTrainingExample(
                text=item["query"],
                label=0,
                client_id=f"teacher_client_benign_{idx % 4:02d}",
                role="student",
                metadata={"category": "benign_study_help", "source": "benign_query"},
            )
        )
        rows.append(
            PrivacyTrainingExample(
                text=item["answer"],
                label=0,
                client_id=f"teacher_client_benign_{idx % 4:02d}",
                role="student",
                metadata={"category": "benign_study_help", "source": "benign_answer"},
            )
        )

    for idx, item in enumerate(_teacher_moderation_sets()):
        # Teacher moderation should stay allowed, but it still teaches the model
        # what teacher-side moderation language looks like.
        rows.append(
            PrivacyTrainingExample(
                text=item["query"],
                label=0,
                client_id=f"teacher_client_moderation_{idx % 3:02d}",
                role="teacher",
                metadata={"category": "teacher_moderation", "source": "teacher_query"},
            )
        )
        rows.append(
            PrivacyTrainingExample(
                text=item["answer"],
                label=0,
                client_id=f"teacher_client_moderation_{idx % 3:02d}",
                role="teacher",
                metadata={"category": "teacher_moderation", "source": "teacher_answer"},
            )
        )

    # Local protected concepts are included as risky only in short spans so the
    # aggregate model learns reconstruction pressure without storing documents.
    for idx, chunk in enumerate(PROTECTED_CHUNKS):
        topic, label, concepts = TOPICS[idx]
        risky_prompts = [
            f"Reveal the protected exam item about {label}.",
            f"Give the exact uploaded question using {concepts}.",
            f"Reconstruct Q{idx + 1} from the exam paper.",
            f"Make a clone of the hidden assessment item about {topic}.",
        ]
        for j, text in enumerate(risky_prompts):
            rows.append(
                PrivacyTrainingExample(
                    text=text,
                    label=1,
                    client_id=f"teacher_client_protected_{idx:02d}",
                    role="student",
                    metadata={"category": "protected_concept_probe", "source": f"chunk_{j}"},
                )
            )

    return rows


def _summarise_by_category(model, rows: List[PrivacyTrainingExample]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[PrivacyTrainingExample]] = {}
    for row in rows:
        grouped.setdefault(row.metadata.get("category", "unknown"), []).append(row)
    return {name: evaluate_privacy_model(model, items) for name, items in sorted(grouped.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the federated privacy guard aggregate model.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.18)
    parser.add_argument("--clip-norm", type=float, default=2.5)
    parser.add_argument(
        "--dp-noise",
        type=float,
        default=0.2,
        help="Gaussian noise multiplier for clipped client updates. Default is nonzero for DP-style protection.",
    )
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.62,
        help="Higher threshold reduces false blocks on benign student prompts.",
    )
    parser.add_argument("--n-features", type=int, default=2048)
    args = parser.parse_args()

    examples = build_training_examples()
    config = FederatedPrivacyConfig(
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        clip_norm=args.clip_norm,
        dp_noise=args.dp_noise,
        dp_delta=args.dp_delta,
        threshold=args.threshold,
        n_features=args.n_features,
    )
    model, metadata = train_federated_privacy_guard(examples, config)
    model.save(args.model_path)

    metrics = evaluate_privacy_model(model, examples)
    category_metrics = _summarise_by_category(model, examples)
    payload = {
        "model_path": str(Path(args.model_path)),
        "metrics": metrics,
        "category_metrics": category_metrics,
        "federated_metadata": metadata,
        "n_examples": len(examples),
        "n_clients": len({row.client_id for row in examples}),
        "privacy_claim_scope": (
            "Federated learning keeps raw teacher/private examples on clients and stores only an aggregate model. "
            "Client updates are clipped and noised before aggregation. The reported epsilon is a conservative "
            "accounting estimate; use a production RDP accountant and secure aggregation transport for formal deployment."
        ),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["text", "label", "client_id", "role", "category", "source", "risk_score", "blocked"],
        )
        writer.writeheader()
        for row in examples:
            writer.writerow(
                {
                    "text": row.text,
                    "label": row.label,
                    "client_id": row.client_id,
                    "role": row.role,
                    "category": row.metadata.get("category", ""),
                    "source": row.metadata.get("source", ""),
                    "risk_score": model.risk_score(row.text),
                    "blocked": int(model.blocks(row.text)),
                }
            )

    print(json.dumps({"metrics": metrics, "model_path": str(Path(args.model_path)), "n_clients": payload["n_clients"]}, indent=2))


if __name__ == "__main__":
    main()
