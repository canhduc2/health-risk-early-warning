"""Train LightGBM trên DỮ LIỆU THẬT NHANES (CDC) và đưa vào sản xuất.

Chạy:
    python scripts/train_nhanes.py

Output (model sản xuất):
    data/models/risk_lgbm_real.joblib + risk_lgbm_real_meta.json
    (feature_names, feature_medians dùng để fill khi inference thiếu chỉ số)

Đặc trưng trùng schema hệ thống: systolic_bp, diastolic_bp, heart_rate,
glucose_fasting, hba1c, creatinine, bmi. Nhãn: tăng huyết áp HOẶC đái tháo
đường (xác định bằng ngưỡng lâm sàng / thuốc đang dùng).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import Config
from src.tier3_risk.ml.risk_model import RiskModel

FEATURES = [
    "systolic_bp", "diastolic_bp", "heart_rate",
    "glucose_fasting", "hba1c", "creatinine", "bmi",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM trên NHANES")
    parser.add_argument("--data", type=str, default="data/datasets/nhanes_2017_2018.csv")
    parser.add_argument("--out", type=str, default="data/models/risk_lgbm_real")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config()
    df = pd.read_csv(args.data)
    df = df.dropna(subset=["label"])
    X, y = df[FEATURES], df["label"].to_numpy(dtype=int)
    print(f"NHANES: n={len(df)} | positive={int(y.sum())} ({y.mean():.1%})")

    # --- Đánh giá khách quan: Stratified 5-fold CV -------------------------
    # C12: imputer fit TRONG từng fold (trên X[tr]) chứ không trên toàn bộ X
    # trước khi chia — tránh leakage metadata imputer từ test vào train.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    aucs, auprcs = [], []
    for tr, te in skf.split(X, y):
        imp = SimpleImputer(strategy="median")
        X_tr_imp = imp.fit_transform(X.iloc[tr])
        X_te_imp = imp.transform(X.iloc[te])
        m = RiskModel(cfg)
        m.train(pd.DataFrame(X_tr_imp, columns=FEATURES), y[tr])
        p = m.predict_proba_score(pd.DataFrame(X_te_imp, columns=FEATURES))
        aucs.append(roc_auc_score(y[te], p))
        auprcs.append(average_precision_score(y[te], p))
    print(f"CV 5-fold: AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f} | "
          f"AUPRC = {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}")

    # --- Model sản xuất: train toàn bộ --------------------------------------
    # Imputer fit trên toàn bộ dữ liệu sẵn có (production OK: giai đoạn này
    # không còn chia train/test để ước lượng bằng chứng độc lập).
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    model = RiskModel(cfg)
    model.train(pd.DataFrame(X_imp, columns=FEATURES), y)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out.with_suffix(".joblib"))
    medians = {c: float(np.nanmedian(df[c])) for c in FEATURES}
    meta = {
        "feature_names": FEATURES,
        "feature_medians": medians,
        "label": "tăng huyết áp HOẶC đái tháo đường (NHANES)",
        "n": int(len(df)),
        "positive": int(y.sum()),
        "auc_cv": float(np.mean(aucs)),
        "auc_cv_std": float(np.std(aucs)),
        "auprc_cv": float(np.mean(auprcs)),
        "data_source": "NHANES (CDC/NCHS): " + ", ".join(df.get("cycle", pd.Series()).unique()) if "cycle" in df.columns else "NHANES (CDC/NCHS)",
        "trained_at": pd.Timestamp.now().isoformat(),
    }
    (out.parent / f"{out.name}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    imp_series = pd.Series(model.model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("Top đặc trưng:")
    for k, v in imp_series.items():
        print(f"  {k:<16} {v}")
    print(f"Đã lưu model sản xuất: {out}.joblib (+ meta)")


if __name__ == "__main__":
    main()
