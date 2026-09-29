"""Điều phối pipeline 3 tầng — điểm gọi duy nhất cho mọi client.

api.py, main.py, chat/agent.py đều gọi assess_patient() ở đây. Module này
KHÔNG chứa logic tầng, chỉ phối hợp theo đúng thứ tự:
    Tầng 1 (stat) → Tầng 2 (rule) → Tầng 3 (fusion + ML) → Tầng 4 (tùy chọn)
"""
from __future__ import annotations

import pandas as pd

from src.config import Config
from src.core.types import AnomalyRecord
from src.data.features import build_feature_matrix
from src.data.preprocess import build_baseline, impute_missing, resample_to_daily
from src.tier1_anomaly.detector import run_tier1, tier1_summary
from src.tier2_knowledge.rules import KnowledgeBase
from src.tier3_risk.ml.risk_model import RiskModel
from src.tier3_risk.report import render_json, render_markdown, save_report
from src.tier3_risk.scoring import RiskResult, RiskScorer

VALUE_COLUMNS = [
    "systolic_bp", "diastolic_bp", "heart_rate", "glucose",
    "glucose_fasting", "hba1c", "creatinine", "egfr", "spo2", "bmi",
]

# Nạp mô hình ML một lần (lazy singleton). Không có model -> ml_score = None.
_ML_MODEL: RiskModel | None = None
_ML_TRIED = False


def _load_joblib(path):
    import joblib

    return joblib.load(path)


def load_ml_model(config: Config) -> RiskModel | None:
    """Nạp LightGBM đã huấn luyện (dữ liệu/models/risk_lgbm.joblib)."""
    global _ML_MODEL, _ML_TRIED
    if _ML_TRIED:
        return _ML_MODEL
    _ML_TRIED = True
    path = config.ml_model_path
    if not path.exists():
        return None
    try:
        _ML_MODEL = RiskModel.load(path, config)
    except Exception:
        _ML_MODEL = None
    return _ML_MODEL


def _ml_score_for(df: pd.DataFrame, value_cols: list[str], config: Config) -> float | None:
    """Điểm mô hình (model score) từ LightGBM trên đặc trưng chuỗi hiện tại.

    Đây là đầu ra THÔ của mô hình sản xuất, không qua calibrator: calibrator
    hiện không được áp dụng ở runtime vì model và calibrator không tạo thành
    một cặp tái lập được (claim C3). Không diễn giải giá trị này như xác suất
    biến cố lâm sàng đã hiệu chỉnh.
    """
    if not config.use_ml or len(df) < config.ml_min_days:
        return None
    model = load_ml_model(config)
    if model is None:
        return None
    fm = build_feature_matrix(df, config, value_cols=value_cols)
    fm = fm.drop(columns=[c for c in ("timestamp", "patient_id") if c in fm.columns])
    if fm.empty:
        return None
    return float(model.predict_features(fm.tail(1)))


def assess_patient(
    df_wide: pd.DataFrame,
    config: Config,
    scorer: RiskScorer,
    explainer=None,
    modes: list[str] | None = None,
) -> dict:
    """Chạy 3 tầng cho một bệnh nhân (wide format), tầng 4 (nếu có) sinh lời giải.

    explainer: đối tượng có phương thức explain(context) -> str (xem
    src/tier4_explain/base.py). Tuỳ chọn — hệ thống chạy đầy đủ khi vắng mặt.
    modes: chế độ chẩn đoán chuyên biệt (htn/dm/ckd/...), xem scoring.py.
    """
    # 1) Làm sạch & đường cơ sở cá nhân
    df = resample_to_daily(df_wide)
    df = impute_missing(df, config)
    value_cols = [c for c in VALUE_COLUMNS if c in df.columns]
    if df.empty or not value_cols:
        return {"risk_level": "INSUFFICIENT_DATA", "message": "Không có chỉ số nào."}

    # Snapshot giá trị hiện tại (dòng cuối) — Tầng 2 luôn kích hoạt luật theo
    # giá trị hiện tại, kể cả khi chưa có lịch sử (mới chỉ 1 lần khám).
    last = df.iloc[-1]
    snapshot = {c: float(last[c]) for c in value_cols if not pd.isna(last[c])}
    if not snapshot:
        return {"risk_level": "INSUFFICIENT_DATA", "message": "Không đủ dữ liệu giá trị."}

    # 2) Tầng 1 — bất thường cá nhân hóa (cần >= min_points; nếu thiếu thì
    #    bỏ qua, hệ thống vẫn đánh giá bằng tri thức y khoa).
    records: list[AnomalyRecord] = []
    tier1_note: str | None = None
    if len(df) >= config.min_points:
        df = build_baseline(df, config.zscore_window_days)
        records = run_tier1(df, value_cols, config)
    else:
        tier1_note = "Chưa đủ lịch sử để phân tích chuỗi thời gian (cần ≥ 7 điểm); đánh giá theo tri thức y khoa."

    # 3) Tầng 2 — tri thức y khoa (luôn chạy từ snapshot giá trị hiện tại)
    # 4) Tầng 3 — tổng hợp rủi ro (thống kê + tri thức + ML + xu hướng)
    ml_score = _ml_score_for(df, value_cols, config)
    result = scorer.score(records, ml_score=ml_score, snapshot=snapshot, modes=modes)

    output = {
        "tier1_summary": tier1_summary(records),
        "tier1_note": tier1_note,
        "ml_score": round(ml_score, 4) if ml_score is not None else None,
        **result.to_dict(),
    }

    # 5) Tầng 4 — giải thích tự nhiên (tùy chọn, không ảnh hưởng kết quả)
    if explainer is not None:
        context = {
            "ml_risk": ml_score,
            "stat_anomalies": [r.to_dict() for r in records],
            "final_risk": result.risk_score,
            "level": result.risk_level,
            "features": snapshot,
            "rules": [h for h in getattr(result, "evidence", [])],
        }
        try:
            output["natural_explanation"] = explainer.explain(context)
        except Exception as e:  # tầng 4 lỗi không được phá luồng chính
            output["natural_explanation"] = None
            output["tier4_note"] = f"Tầng 4 không sinh được lời giải: {e}"

    return output


__all__ = [
    "VALUE_COLUMNS",
    "RiskResult",
    "load_ml_model",
    "assess_patient",
    "save_report",
    "render_json",
    "render_markdown",
]
