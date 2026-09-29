"""Bootstrap 95% CI cho ROC-AUC test (LR vs LGBM) — EXP-TEMPORAL-LMF.

Dữ liệu: roc_predictions.json.gz (y_true/y_score của split temporal).
Phương pháp: percentile bootstrap trên index bệnh nhân, 2000 lần lặp, seed 42.
Ghi ra bootstrap_ci.json — bổ sung cho claim C8 (ước lượng bất định).
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "experiments" / "EXP-TEMPORAL-LMF"

N_BOOT = 2000
SEED = 42


def main() -> None:
    with gzip.open(SRC / "roc_predictions.json.gz", "rt", encoding="utf-8") as f:
        data = json.load(f)

    rng = np.random.default_rng(SEED)
    n = len(data["lr"]["y_true"])
    y_true = np.asarray(data["lr"]["y_true"])
    rows = np.arange(n)

    result = {"n": n, "n_boot": N_BOOT, "seed": SEED, "method": "percentile bootstrap"}
    for model in ("lr", "lgbm"):
        y_score = np.asarray(data[model]["y_score"])
        auc_obs = roc_auc_score(y_true, y_score)
        aucs = []
        for _ in range(N_BOOT):
            idx = rng.choice(rows, size=n, replace=True)
            if len(np.unique(y_true[idx])) < 2:
                continue
            aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
        aucs = np.asarray(aucs)
        lo, hi = np.percentile(aucs, [2.5, 97.5])
        result[model] = {
            "auc": round(float(auc_obs), 4),
            "ci95_lo": round(float(lo), 4),
            "ci95_hi": round(float(hi), 4),
            "sd": round(float(aucs.std(ddof=1)), 4),
        }
        print(f"{model.upper()}: AUC={auc_obs:.4f} 95%CI [{lo:.4f}, {hi:.4f}]")

    # Kiểm tra xem khoảng CI của hai mô hình có tách nhau không.
    lr, lg = result["lr"], result["lgbm"]
    overlap = not (lr["ci95_hi"] < lg["ci95_lo"] or lg["ci95_hi"] < lr["ci95_lo"])
    result["interpretation"] = (
        "overlapping" if overlap else "non-overlapping"
    )
    print(f"Interpretation: {result['interpretation']} (95% CI)")
    (SRC / "bootstrap_ci.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Saved:", SRC / "bootstrap_ci.json")


if __name__ == "__main__":
    sys.exit(main())