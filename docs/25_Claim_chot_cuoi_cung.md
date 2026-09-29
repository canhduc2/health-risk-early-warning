# 25. Bản claim cuối cùng (chốt cho revision)

> Nguồn sự thật duy nhất cho vòng sửa đổi theo phản biện. Mọi wording trong
> manuscript, README, docs và code comment **phải** khớp bảng dưới đây.
> Bất kỳ chỗ nào còn claim mạnh hơn bảng này = bug cần sửa.

## Nguyên tắc chốt

1. **Giữ claim** khi implementation + experiment hiện tại chứng minh được.
2. **Hạ claim** khi chưa có bằng chứng — không tạo experiment giả để giữ wording.
3. Chỉ các fix **correctness/safety cần thiết** mới sửa code; phần còn lại chỉ sửa manuscript.
4. Sau mỗi fix correctness ảnh hưởng số liệu → mới chạy experiment bị ảnh hưởng.

---

## Bảng quyết định tổng quát

| # | Claim cũ | Verdict | Claim mới (chốt) | Code | Experiment |
|---|----------|---------|------------------|------|------------|
| C1 | "Đánh giá khung tích hợp" | HẠ | "Thiết kế, triển khai và đánh giá kỹ thuật các **thành phần** của khung tích hợp" | — | — |
| C2 | Missing component → 0 | SỬA | `INSUFFICIENT_DATA` khi thiếu bằng chứng bắt buộc | `scoring.py` | Chạy lại test; artifact R chỉ nếu R nằm trong báo cáo |
| C3 | ML output = calibrated probability | HẠ + SỬA | Disable calibrator runtime; gọi "điểm mô hình (model score)" | `pipeline.py` | — |
| C4 | 9 "clinical rules" | HẠ | 9 "rule prototypes"; rule chưa approved/mismatch → Draft/demo | `knowledge_base.json` | — |
| C5 | Tầng luật độc lập Tier 1 | SỬA | Rule engine **luôn** evaluate full snapshot; records chỉ bổ sung evidence | `scoring.py` | Test regression |
| C6 | "Early-warning/online" Tier 1 | HẠ | "Phân tích hồi cứu (retrospective prototype)" | — | — |
| C7 | MIMIC = temporal/external validation | HẠ | "Stress test theo shifted-year partition trên stored artifact" | — | — |
| C8 | LR vượt LGBM | HẠ | "Point estimates gợi ý; chưa kết luận khác biệt có ý nghĩa (thiếu CI)" | — | ⚠️ Bootstrap CI nếu đủ thời gian |
| C9 | Isotonic LMF = calibration | HẠ | "Phân tích thăm dò; không khẳng định xác suất đã hiệu chuẩn" | — | — |
| C10 | NHANES AUC ≈ 0.94 = dự báo nguy cơ | HẠ | "Đối sánh phân loại nhãn cắt ngang có feature–label circularity đã biết" | — | — |
| C11 | "Trọng số tối ưu/Bayesian" | HẠ | "Trọng số là design parameters; sensitivity là technical, không phải clinical validation" | — | — |
| C12 | Production `auc_cv` = bằng chứng | HẠ | Không dùng metadata làm validation; by chứng lấy từ protocol 70/15/15 | — | — |

---

## C1 — Framework-level evaluation

### Quyết định: HẠ (không làm framework-level experiment ở vòng này)

Chưa có outcome hợp lệ cho `risk_score R` và không có ablation S/K/M/T → không
giữ câu gốc.

**Wording chốt (Abstract/Introduction/Conclusion):**

> "Nghiên cứu đóng góp ở thiết kế, triển khai và đánh giá kỹ thuật các thành
> phần của khung tích hợp; nghiên cứu chưa đánh giá hiệu năng dự báo hoặc hiệu
> quả lâm sàng của điểm tổng hợp đa tầng như một endpoint độc lập."

- Code: không sửa.
- Điểm số đa tầng `R` trong báo cáo dùng như **đầu ra vận hành kỹ thuật**, không
  như predictor đối chiếu outcome.

---

## C2 — Missing component → 0

### Quyết định: SỬA CODE (correctness/safety)

Công thức hiện tại biến "thiếu bằng chứng" thành "bằng chứng nguy cơ thấp":

```text
S=T=0, K=M=1  ⇒  R = 0.30·0 + 0.35·1 + 0.25·1 + 0.10·0 = 0.60 < HIGH(0.66)
```

### Hành vi chốt

1. Nếu `ml_score` là `None` (thiếu model) **và** không có records (thiếu Tier 1)
   → `risk_status = INSUFFICIENT_DATA`, kèm `data_sufficiency` chi tiết.
2. Không tự ý gán `0` cho thành phần thiếu khi kéo theo quyết định phân tầng.
3. Ngưỡng 0.33/0.66 chỉ áp dụng cho cấu hình có đủ bằng chứng theo `data_sufficiency` ≥ TRUNG_BINH.

**Ghi chú vào code/manuscript ngay sau công thức fusion:**

> "Các ngưỡng phân tầng chỉ áp dụng khi tập thành phần bắt buộc có đủ bằng
> chứng theo cấu hình đã định; không dùng chung ngưỡng khi thiếu Tầng 1/ML nếu
> chưa được hiệu chỉnh riêng."

- Files: `src/tier3_risk/scoring.py`, cấu hình/schema response.
- Test: all components / only S missing / only T missing / ML missing / S+T
  missing / K=M max nhưng thiếu S,T / threshold không tự gọi HIGH.

---

## C3 — Runtime calibration mismatch

### Quyết định: HẠ + SỬA CODE

Không có paired artifact (model + calibrator + split). Calibrator hiện tại lấy
từ `EXP-ML-LGBM-42`, model sản xuất là `risk_lgbm_real` (khác class_weight,
khác training) → **disable calibrator runtime**.

**Wording chốt:**

> "Trong phiên bản đánh giá này, đầu ra của mô hình sản xuất được gọi là *điểm
> mô hình* (model score), không phải xác suất đã hiệu chỉnh; calibrator chỉ được
> áp dụng khi model và calibrator được huấn luyện/đóng gói như một cặp tái lập được."

- Code: `src/core/pipeline.py` bỏ nhánh `apply_calibration`.
- Experiment: không cần chạy lại (calibration không nằm trong kết quả chính).

---

## C4 — Rule prototypes chưa approved

### Quyết định: HẠ + SỬA DATA

9 luật đang `status:"active"` nhưng `approved_by/approved_at = null`; một số rule
mismatch guideline (R_END_01, R_CV_03, R_KID_02, ...).

**Hành động chốt:**

- Rules chưa review hoặc còn mismatch → `status: "draft"` / `"review"` (demo mode).
- `approved_by/approved_at` bắt buộc khác `null` mới được `active`.

**Wording chốt:**

> "Các luật hiện tại là rule prototypes phục vụ kiểm thử kỹ thuật; các luật có
> khác biệt với guideline hoặc chưa có phê duyệt chuyên môn không được sử dụng
> để hỗ trợ quyết định lâm sàng."

- Files: `src/tier2_knowledge/knowledge_base.json`.
- Mỗi rule cần: guideline, page/section, excerpt, version, review status.

---

## C5 — Rule engine phụ thuộc Tier 1 records

### Quyết định: SỬA CODE (architectural correctness)

Hiện tại `scoring.py:284-286`:

```python
hits = self.kb.evaluate_from_records(records, modes=modes) if records else []
if snapshot and not records:
    hits = self.kb.evaluate(snapshot, modes=modes)
```

→ khi có records, rule engine bỏ qua metric hiện tại không có AnomalyRecord.

**Logic chốt:**

```python
hits = self.kb.evaluate(snapshot, modes=modes)   # luôn đánh giá full snapshot
# records chỉ dùng để bổ sung bằng chứng đường cơ sở/xu hướng
```

- Test regression: metric không có z-score/record nhưng vượt rule threshold → rule
  vẫn trigger.
- Experiment: nếu rule output ảnh hưởng `R` được báo cáo → regenerate artifact.

---

## C6 — Tier 1 look-ahead leakage

### Quyết định: HẠ (giữ implementation, giới hạn claim)

`preprocess.py:52-57` dùng `np.interp` toàn chuỗi + `ffill().bfill()` (tương
lai); `build_baseline` rolling chưa `shift(1)` → có future information.

**Wording chốt:**

> "Tầng 1 được đánh giá hồi cứu (retrospective prototype) trên toàn bộ chuỗi;
> nghiên cứu chưa cung cấp bằng chứng cho vận hành cảnh báo trực tuyến tại thời
> điểm t."

- Không dùng từ: online / real-time / early-warning / causal anomaly detection.
- Nếu vòng sau muốn giữ online → Route B: shift(1), bỏ bfill, rerun experiment.

---

## C7 — MIMIC-IV reproducibility

### Quyết định: HẠ

`build_mimic_dataset.py`/`run_temporal_mimiciv.py` bị gitignore (`*_mimic*`),
chưa từng commit → main không tái lập được pipeline. Ngày `shifted year` cũng
không phải lịch chung.

**Wording chốt:**

> "Nhánh MIMIC-IV là phân tích stress test theo shifted-year partition trên các
> artifact đã lưu; prediction time, label boundary và patient-level split chưa
> được tái lập độc lập nên không được gọi là external/temporal validation."

- Không dùng: temporal validation / external validation / prediction validation
  cho nhánh MIMIC.

---

## C8 — LMF: LR vs LGBM

### Quyết định: HẠ (thêm CI nếu đủ thời gian)

LMF chỉ có 49 events (train) / 67 (test); LGBM train `ROC-AUC=1.0, AUPRC=1.0`.

**Wording chốt:**

> "Các point estimate gợi ý LR phân biệt tốt hơn trên test (LR 0,8209; LGBM
> 0,7709); chưa kết luận khác biệt có ý nghĩa do chưa có ước lượng bất định.
> Mô hình boosting cho AUC/AUPRC đạt mức bão hòa trên tập huấn luyện — bằng
> chứng overfit rõ ràng trên miền sự kiện thưa."

- Experiment (optional P1): bootstrap 95% CI cho ROC-AUC / PR-AUC / ΔLR−LGBM;
  báo train performance kèm cảnh báo overfit.

---

## C9 — LMF isotonic in-sample

### Quyết định: HẠ (không giữ làm main evidence)

Isotonic fit trên train predictions (`run_temporal_validation.py:167-168`),
LGBM test sau isotonic `roc_auc=0.5054`.

**Wording chốt:**

> "Hiệu chỉnh isotonic của nhánh LMF là phân tích thăm dò được khớp nội bộ
> (in-sample) và không được dùng để khẳng định xác suất đã hiệu chuẩn."

- Muốn giữ calibrated probability → OOF calibration + calibration set riêng (P1).

---

## C10 — NHANES feature–label circularity

### Verdict: Giữ cảnh báo hiện tại, không dùng cho claim dự báo.

- Tiêu đề mục có thể đổi thành: "Đối sánh phân loại nhãn cắt ngang có
  feature–label circularity đã biết".
- Không dùng AUC ≈ 0.94 cho: clinical prediction / early detection / risk prediction.

---

## C11 — Trọng số fusion

### Wording chốt:

> "Trọng số {0,30; 0,35; 0,25; 0,10} là design parameters của nguyên mẫu;
> sensitivity analysis hiện có là kiểm thử kỹ thuật, không phải clinical
> validation."

- Không dùng: optimal weights / validated weights / Bayesian fusion.

---

## C12 — Production metadata

### Hành động:

- `scripts/train_nhanes.py` fit imputer trước StratifiedKFold (leakage) → dùng
  `Pipeline` để imputer fit trong từng fold, hoặc tuyên bố `auc_cv` không phải
  bằng chứng.
- **Wording chốt:** "Metadata huấn luyện model sản xuất không được dùng làm
  estimate đánh giá độc lập; bằng chứng benchmark lấy từ protocol train/val/test
  riêng."
- Không tái sinh metadata trừ khi thực sự dùng lại.

---

## Thứ tự thực thi

```text
Phase 0  Freeze baseline (commit hiện tại) + copy bảng claim này
Phase 1  Sửa manuscript theo C1, C3, C4, C6, C7, C8, C9, C10, C11   (chỉ MD, không code)
Phase 2  Sửa code correctness theo C2, C3, C5 (+ C4 data, C12 pipeline)
Phase 3  Unit/integration test cho các fix trên
Phase 4  Experiment tùy chọn: C8 bootstrap CI; C9 OOF calibration
Phase 5  Đồng bộ README/docs/comment theo bảng claim
Phase 6  Final audit (claim không vượt bằng chứng; repo tái lập)
```

## Definition of Done

- [ ] Không còn claim "đánh giá khung tích hợp".
- [ ] Không còn missing→0 trong phân tầng; có `INSUFFICIENT_DATA`.
- [ ] Runtime không áp calibrator lạ.
- [ ] Mọi rule hoặc approved hoặc Draft.
- [ ] Rule engine luôn evaluate full snapshot.
- [ ] Tier 1 gọi là retrospective prototype.
- [ ] MIMIC gọi là shifted-year stress test trên artifact.
- [ ] LMF: point estimates + cảnh báo overfit, không kết luận so sánh.
- [ ] Isotonic LMF = exploratory.
- [ ] NHANES circularity được cảnh báo.
- [ ] Trọng số = design parameters.
- [ ] README đồng bộ manuscript.